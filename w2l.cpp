#include <iostream>
#include <stdlib.h>
#include <string>
#include <typeinfo>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Dictionary.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/W2lDataset.h"
#include "module/module.h"
#include "runtime/Logger.h"
#include "runtime/Serial.h"
#include "decoder/Decoder.h"
#include "decoder/Utils.h"
#include "decoder/Trie.h"
#include "decoder/WordLMDecoder.h"
#include "decoder/TokenLMDecoder.h"
#include "lm/KenLM.h"

#include "w2l.h"

#include "simpledecoder.cpp"

using namespace w2l;

class EngineBase {
public:
    int numClasses;
    std::unordered_map<std::string, std::string> config;
    std::shared_ptr<fl::Module> network;
    std::shared_ptr<SequenceCriterion> criterion;
    std::string criterionType;
    Dictionary tokenDict;
};

class Emission {
public:
    Emission(fl::Variable emission) {
        this->emission = emission;
    }
    ~Emission() {}

    char *text(EngineBase *engine) {
        auto tokenPrediction =
            afToVector<int>(engine->criterion->viterbiPath(emission.array()));
        auto letters = tknPrediction2Ltr(tokenPrediction, engine->tokenDict);
        if (letters.size() > 0) {
            std::ostringstream ss;
            for (auto s : letters) ss << s;
            return strdup(ss.str().c_str());
        }
        return strdup("");
    }

    fl::Variable emission;
};

class Engine : public EngineBase {
public:
    Engine(const char *acousticModelPath, const char *tokensPath) {
        // TODO: set criterionType "correctly"
        W2lSerializer::load(acousticModelPath, config, network, criterion);
        auto flags = config.find(kGflags);
        // loading flags globally like this is gross, only way to work around it will be parameterizing everything about wav2letter better
        gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

        criterionType = FLAGS_criterion;
        network->eval();
        criterion->eval();

        tokenDict = Dictionary(tokensPath);
        numClasses = tokenDict.indexSize();
    }
    ~Engine() {}

    Emission *process(float *samples, size_t sample_count) {
        struct W2lLoaderData data = {};
        std::copy(samples, samples + sample_count, std::back_inserter(data.input));

        auto feat = featurize({data}, {});
        auto input = af::array(feat.inputDims, feat.input.data());
        auto rawEmission = network->forward({fl::input(input)}).front();
        return new Emission(rawEmission);
    }

    bool exportModel(const char *path) {
        std::ofstream outfile;
        outfile.open(path, std::ios::out | std::ios::binary);
        if (!outfile.is_open()) {
                std::cout << "[w2lapi] error, could not open file '" << path << "' (aborting export)" << std::endl;
            return false;
        }

        auto seq = dynamic_cast<fl::Sequential *>(network.get());
        for (auto &module : seq->modules()) {
            if (!exportLayer(outfile, module.get())) {
                std::cout << "[w2lapi] aborting export" << std::endl;
                return false;
            }
        }
        return true;
    }

private:
    std::tuple<int, int> splitOn(std::string s, std::string on) {
        auto split = s.find(on);
        auto first = s.substr(0, split);
        auto second = s.substr(split + on.size());
        // std::cout << "string [" << s << "] on [" << on << "] first " << first << " second " << second << std::endl;
        return {std::stoi(first), std::stoi(second)};
    }

    std::string findParens(std::string s) {
        auto start = s.find('(');
        auto end = s.find(')', start);
        auto sp = s.substr(start + 1, end - start - 1);
        // std::cout << "string split [" << s << "] " << start << " " << end << " [" << sp << "]" << std::endl;
        return sp;
    }

    void exportParams(std::ofstream& f, fl::Variable params) {
        auto array = afToVector<float>(params.array());
        for (auto& p : array) {
            f << std::hex << (uint32_t&)p;
            if (&p != &array.back()) {
                f << " ";
            }
        }
        f << std::dec;
    }

    bool exportLayer(std::ofstream& f, fl::Module *module) {
        auto pretty = module->prettyString();
        auto type = pretty.substr(0, pretty.find(' '));
        std::cout << "[w2lapi] exporting: " << pretty << std::endl;
        if (type == "WeightNorm") {
//            auto wn = dynamic_cast<fl::WeightNorm *>(module);
//            auto lastParam = pretty.rfind(",") + 2;
//            auto dim = pretty.substr(lastParam, pretty.size() - lastParam - 1);
//            f << "WN " << dim << " ";
//            exportLayer(f, wn->module().get());
        } else if (type == "View") {
            auto ratio = findParens(pretty);
            f << "V " << findParens(pretty) << "\n";
        } else if (type == "Dropout") {
            f << "DO " << findParens(pretty) << "\n";
        } else if (type == "Reorder") {
            auto dims = findParens(pretty);
            std::replace(dims.begin(), dims.end(), ',', ' ');
            f << "RO " << dims << "\n";
        } else if (type == "GatedLinearUnit") {
            f << "GLU " << findParens(pretty) << "\n";
        } else if (type == "Conv2D") {
            // Conv2D (234->514, 23x1, 1,1, 0,0, 1, 1) (with bias)
            auto parens = findParens(pretty);
            bool bias = pretty.find("with bias") >= 0;
            int inputs, outputs, szX, szY, padX, padY, strideX, strideY, dilateX, dilateY;
            // TODO: I could get some of these from the params' dims instead of string parsing...

            auto comma1 = parens.find(',') + 1;
            std::tie(inputs, outputs) = splitOn(parens.substr(0, comma1), "->");

            auto comma2 = parens.find(',', comma1) + 1;
            std::tie(szX, szY) = splitOn(parens.substr(comma1, comma2 - comma1 - 1), "x");

            auto comma4 = parens.find(',', parens.find(',', comma2) + 1) + 1;
            std::tie(strideX, strideY) = splitOn(parens.substr(comma2, comma4 - comma2 - 1), ",");

            auto comma6 = parens.find(',', parens.find(',', comma4) + 1) + 1;
            std::tie(padX, padY) = splitOn(parens.substr(comma4, comma6 - comma4 - 1), ",");

            auto comma8 = parens.find(',', parens.find(',', comma6) + 1) + 1;
            std::tie(dilateX, dilateY) = splitOn(parens.substr(comma6, comma8 - comma6 - 1), ",");

            // FIXME we're ignoring everything after padX because I don't know the actual spec
            // string split [Conv2D (40->200, 13x1, 1,1, 170,0, 1, 1) (with bias)] 7 39 [40->200, 13x1, 1,1, 170,0, 1, 1]
            // fl::Conv2D C2 [inputChannels] [outputChannels] [xFilterSz] [yFilterSz] [xStride] [yStride] [xPadding <OPTIONAL>] [yPadding <OPTIONAL>] [xDilation <OPTIONAL>] [yDilation <OPTIONAL>]
            f << "C " << inputs << " " << outputs << " " << szX << " " << szY << " " << padX << " | ";
            exportParams(f, module->param(0));
            if (bias) {
                f << " | ";
                exportParams(f, module->param(1));
            }
            f << "\n";
        } else if (type == "Linear") {
            int inputs, outputs;
            std::tie(inputs, outputs) = splitOn(findParens(pretty), "->");
            f << "L " << inputs << " " << outputs << " | ";
            exportParams(f, module->param(0));
            f << "\n";
        } else {
            // TODO: also write error to the file?
            std::cout << "[w2lapi] error, unknown layer type: " << type << std::endl;
            return false;
        }
        return true;
    }
};

DecoderOptions toW2lDecoderOptions(const w2l_decode_options &opts) {
    return DecoderOptions(
                opts.beamsize,
                opts.beamthresh,
                opts.lmweight,
                opts.wordscore,
                opts.unkweight,
                opts.logadd,
                opts.silweight,
                CriterionType::ASG);
}

class WrapDecoder {
public:
    WrapDecoder(Engine *engine, const char *languageModelPath, const char *lexiconPath, const w2l_decode_options *opts) {
        tokenDict = engine->tokenDict;
        silIdx = tokenDict.getIndex(kSilToken);

        auto lexicon = loadWords(lexiconPath, -1);
        wordDict = createWordDict(lexicon);
        lm = std::make_shared<KenLM>(languageModelPath, wordDict);

        // Load the trie: either the flattened cache from disk, or recreate from the lexicon
        std::string flatTriePath = std::string(lexiconPath) + ".flattrie";
        std::ifstream flatTrieIn(flatTriePath.c_str());
        if (!flatTrieIn.good()) {
            // taken from Decode.cpp
            // Build Trie
            int blankIdx = engine->criterionType == kCtcCriterion ? engine->tokenDict.getIndex(kBlankToken) : -1;
            std::shared_ptr<Trie> trie = std::make_shared<Trie>(engine->tokenDict.indexSize(), silIdx);
            auto startState = lm->start(false);
            for (auto& it : lexicon) {
                const std::string& word = it.first;
                int usrIdx = wordDict.getIndex(word);
                float score = -1;
                // if (FLAGS_decodertype == "wrd") {
                if (true) {
                    LMStatePtr dummyState;
                    std::tie(dummyState, score) = lm->score(startState, usrIdx);
                }
                for (auto& tokens : it.second) {
                    auto tokensTensor = tkn2Idx(tokens, engine->tokenDict);
                    trie->insert(tokensTensor, usrIdx, score);
                }
            }

            // Smearing
            // TODO: smear mode argument?
            SmearingMode smear_mode = SmearingMode::MAX;
            /*
            SmearingMode smear_mode = SmearingMode::NONE;
            if (FLAGS_smearing == "logadd") {
                smear_mode = SmearingMode::LOGADD;
            } else if (FLAGS_smearing == "max") {
                smear_mode = SmearingMode::MAX;
            } else if (FLAGS_smearing != "none") {
                LOG(FATAL) << "[Decoder] Invalid smearing mode: " << FLAGS_smearing;
            }
            */
            trie->smear(smear_mode);

            flatTrie = std::make_shared<FlatTrie>(toFlatTrie(trie->getRoot()));
            std::ofstream out(flatTriePath.c_str());
            size_t byteSize = 4 * flatTrie->storage.size();
            out << byteSize;
            out.write(reinterpret_cast<const char*>(flatTrie->storage.data()), byteSize);
        } else {
            flatTrie = std::make_shared<FlatTrie>();
            size_t byteSize;
            flatTrieIn >> byteSize;
            flatTrie->storage.resize(byteSize / 4);
            flatTrieIn.read(reinterpret_cast<char *>(flatTrie->storage.data()), byteSize);
        }

        // the root maxScore should be 0 during search and it's more convenient to set here
        const_cast<FlatTrieNode *>(flatTrie->getRoot())->maxScore = 0;

        CriterionType criterionType = CriterionType::ASG;
        if (engine->criterionType == kCtcCriterion) {
            criterionType = CriterionType::CTC;
        } else if (engine->criterionType != kAsgCriterion) {
            // FIXME:
            LOG(FATAL) << "[Decoder] Invalid model type: " << engine->criterionType;
        }
        decoderOpt = toW2lDecoderOptions(*opts);

        KenFlatTrieLM::LM lmWrap;
        lmWrap.ken = lm;
        lmWrap.trie = flatTrie;

        auto transition = afToVector<float>(engine->criterion->param(0).array());
        decoder.reset(new SimpleDecoder<KenFlatTrieLM::LM, KenFlatTrieLM::State>{
            decoderOpt,
            lmWrap,
            silIdx,
            wordDict.getIndex(kUnkToken),
            transition});
    }
    ~WrapDecoder() {}

    DecodeResult decode(Emission *emission) {
        auto rawEmission = emission->emission;
        auto emissionVec = afToVector<float>(rawEmission);
        int N = rawEmission.dims(0);
        int T = rawEmission.dims(1);

        std::vector<float> score;
        std::vector<std::vector<int>> wordPredictions;
        std::vector<std::vector<int>> letterPredictions;
        KenFlatTrieLM::State startState;
        startState.lex = flatTrie->getRoot();
        startState.kenState = lm->start(0);
        return decoder->normal(emissionVec.data(), T, N, startState);
        //return decoder->groupThreading(emissionVec.data(), T, N);
    }

    char *resultWords(const DecodeResult &result) {
        auto rawWordPrediction = validateIdx(result.words, wordDict.getIndex(kUnkToken));
        auto wordPrediction = wrdIdx2Wrd(rawWordPrediction, wordDict);
        auto words = join(" ", wordPrediction);
        return strdup(words.c_str());
    }

    char *resultTokens(const DecodeResult &result) {
        auto tknIdx = result.tokens;

        // ends with a -1 token, make into silence instead
        // tknIdx2Ltr will filter out the first and last if they are silences
        if (tknIdx.size() > 0 && tknIdx.back() == -1)
           tknIdx.back() = silIdx;

        auto tknLtrs = tknIdx2Ltr(tknIdx, tokenDict);
        std::string out;
        for (const auto &ltr : tknLtrs)
            out.append(ltr);
        return strdup(out.c_str());
    }

    std::shared_ptr<KenLM> lm;
    FlatTriePtr flatTrie;
    std::unique_ptr<SimpleDecoder<KenFlatTrieLM::LM, KenFlatTrieLM::State>> decoder;
    Dictionary wordDict;
    Dictionary tokenDict;
    DecoderOptions decoderOpt;
    int silIdx;
};

namespace DFALM {

std::string tokens = "|'abcdefghijklmnopqrstuvwxyz";
const int TOKENS = 28;
std::vector<uint8_t> charToToken(128);
uint32_t EDGE_INIT[TOKENS] = {0};

enum {
    FLAG_NONE  = 0,
    FLAG_START = 1,
    FLAG_TERM  = 2,
    FLAG_LM    = 4,
};

struct LM {
    const cfg *dfa;
    const cfg *get(const cfg *base, const int32_t idx) const {
        return reinterpret_cast<const cfg *>(reinterpret_cast<const uint8_t *>(base) + idx);
    }
    int wordStartsBefore = 1000000000;
};

struct State {
    const cfg *lex;

    // used for making an unordered_set of const State*
    struct Hash {
        const LM &unused;
        size_t operator()(const State *v) const {
            return std::hash<const void*>()(v->lex);
        }
    };

    struct Equality {
        const LM &unused;
        int operator()(const State *v1, const State *v2) const {
            return v1->lex == v2->lex;
        }
    };

    // Iterate over labels, calling fn with: the new State, the label index and the lm score
    template <typename Fn>
    void forLabels(const LM &lm, Fn&& fn) const {
        const float commandScore = 1.5;
        if (lex->token == 0) { // word boundary at sil token
            fn(*this, reinterpret_cast<const uint8_t*>(lex) - reinterpret_cast<const uint8_t*>(lm.dfa), commandScore);
        }
    }

    // Call finish() on the lm, like for end-of-sentence scoring
    std::pair<State, float> finish(const LM &lm) const {
        return {*this, 0};
    }

    float maxWordScore() const {
        return 0; // could control whether the beam search gets scores before finishing commands
    }

    // Iterate over children of the state, calling fn with:
    // new State, new token index and whether the new state has children
    template <typename Fn>
    void forChildren(int frame, const LM &lm, Fn&& fn) const {
        if (lex->token == 0 && frame >= lm.wordStartsBefore)
            return;
        for (int i = 0; i < lex->nEdges; ++i) {
            auto nidx = lex->edges[i];
            auto nlex = lm.get(lex, nidx);
            fn(State{nlex}, nlex->token, nlex->token != 0);
        }
    }

    State &actualize() {
        return *this;
    }
};

} // namespace DFALM

extern "C" {

typedef struct w2l_engine w2l_engine;
typedef struct w2l_decoder w2l_decoder;
typedef struct w2l_emission w2l_emission;
typedef struct w2l_decoderesult w2l_decoderesult;

w2l_engine *w2l_engine_new(const char *acoustic_model_path, const char *tokens_path) {
    // TODO: what other engine config do I need?
    auto engine = new Engine(acoustic_model_path, tokens_path);
    return reinterpret_cast<w2l_engine *>(engine);
}

w2l_emission *w2l_engine_process(w2l_engine *engine, float *samples, size_t sample_count) {
    auto emission = reinterpret_cast<Engine *>(engine)->process(samples, sample_count);
    return reinterpret_cast<w2l_emission *>(emission);
}

bool w2l_engine_export(w2l_engine *engine, const char *path) {
    return reinterpret_cast<Engine *>(engine)->exportModel(path);
}

void w2l_engine_free(w2l_engine *engine) {
    if (engine)
        delete reinterpret_cast<Engine *>(engine);
}

char *w2l_emission_text(w2l_engine *engine, w2l_emission *emission) {
    return reinterpret_cast<Emission *>(emission)->text(reinterpret_cast<Engine *>(engine));
}

float *w2l_emission_values(w2l_emission *emission, int *frames, int *tokens) {
    auto em = reinterpret_cast<Emission *>(emission);
    auto data = afToVector<float>(em->emission.array());
    *frames = em->emission.array().dims(1);
    *tokens = em->emission.array().dims(0);
    int datasize = sizeof(float) * *frames * *tokens;
    float *out = static_cast<float *>(malloc(datasize));
    memcpy(out, data.data(), datasize);
    return out;
}

void w2l_emission_free(w2l_emission *emission) {
    if (emission)
        delete reinterpret_cast<Emission *>(emission);
}

w2l_decoder *w2l_decoder_new(w2l_engine *engine, const char *kenlm_model_path, const char *lexicon_path, const w2l_decode_options *opts) {
    // TODO: what other config? beam size? smearing? lm weight?
    auto decoder = new WrapDecoder(reinterpret_cast<Engine *>(engine), kenlm_model_path, lexicon_path, opts);
    return reinterpret_cast<w2l_decoder *>(decoder);
}

w2l_decoderesult *w2l_decoder_decode(w2l_decoder *decoder, w2l_emission *emission) {
    auto result = new DecodeResult(reinterpret_cast<WrapDecoder *>(decoder)->decode(reinterpret_cast<Emission *>(emission)));
    return reinterpret_cast<w2l_decoderesult *>(result);
}

char *w2l_decoder_result_words(w2l_decoder *decoder, w2l_decoderesult *decoderesult) {
    auto decoderObj = reinterpret_cast<WrapDecoder *>(decoder);
    auto result = reinterpret_cast<DecodeResult *>(decoderesult);
    return decoderObj->resultWords(*result);
}

char *w2l_decoder_result_tokens(w2l_decoder *decoder, w2l_decoderesult *decoderesult) {
    auto decoderObj = reinterpret_cast<WrapDecoder *>(decoder);
    auto result = reinterpret_cast<DecodeResult *>(decoderesult);
    return decoderObj->resultTokens(*result);
}

void w2l_decoderesult_free(w2l_decoderesult *decoderesult) {
    if (decoderesult)
        delete reinterpret_cast<DecodeResult *>(decoderesult);
}

void w2l_decoder_free(w2l_decoder *decoder) {
    if (decoder)
        delete reinterpret_cast<WrapDecoder *>(decoder);
}

char *w2l_decoder_dfa(w2l_engine *engine, w2l_decoder *decoder, w2l_emission *emission, cfg *dfa, cmd_decode_opts opts) {
    auto engineObj = reinterpret_cast<Engine *>(engine);
    auto decoderObj = reinterpret_cast<WrapDecoder *>(decoder);
    auto rawEmission = reinterpret_cast<Emission *>(emission)->emission;

    auto emissionVec = afToVector<float>(rawEmission);
    int T = rawEmission.dims(0);
    int N = rawEmission.dims(1);
    auto &transitions = decoderObj->decoder->transitions_;

    auto emissionTransmissionAdjustment = [&emissionVec, &transitions, T](const std::vector<int> &tokens, int from, int i) {
        float score = 0;
        if (i > from) {
            score += transitions[tokens[i] * T + tokens[i - 1]];
        } else {
            score += transitions[tokens[i] * T + 0]; // from silence
        }
        score += emissionVec[i * T + tokens[i]];
        return score;
    };

    auto emissionTransmissionScore = [&emissionTransmissionAdjustment, &transitions, T](const std::vector<int> &tokens, int from, int to) {
        float score = 0.0;
        for (int i = from; i < to; ++i) {
            score += emissionTransmissionAdjustment(tokens, from, i);
        }
        score += transitions[0 * T + tokens[to - 1]]; // to silence
        return score;
    };

    auto worstEmissionTransmissionWindowFraction = [&emissionTransmissionAdjustment, &transitions, T](
            const std::vector<int> &tokens1,
            const std::vector<int> &tokens2,
            int from, int to, int window) {
        float score1 = 0.0;
        float score2 = 0.0;
        float worst = INFINITY;
        for (int i = from; i < to; ++i) {
            score1 += emissionTransmissionAdjustment(tokens1, from, i);
            score2 += emissionTransmissionAdjustment(tokens2, from, i);
            if (i < from + window)
                continue;
            if (worst > score1 / score2)
                worst = score1 / score2;
            score1 -= emissionTransmissionAdjustment(tokens1, from, i - window);
            score2 -= emissionTransmissionAdjustment(tokens2, from, i - window);
        }
        score1 += transitions[0 * T + tokens1[to - 1]]; // to silence
        score2 += transitions[0 * T + tokens2[to - 1]]; // to silence
        if (worst > score1 / score2)
            worst = score1 / score2;
        return worst;
    };


    auto tokensToString = [engineObj](const std::vector<int> &tokens, int from, int to) {
        std::string out;
        for (int i = from; i < to; ++i)
            out.append(engineObj->tokenDict.getEntry(tokens[i]));
        return out;
    };
    auto tokensToStringDedup = [engineObj](const std::vector<int> &tokens, int from, int to) {
        std::string out;
        int tok = -1;
        for (int i = from; i < to; ++i) {
            if (tok == tokens[i])
                continue;
            tok = tokens[i];
            out.append(engineObj->tokenDict.getEntry(tok));
        }
        return out;
    };

    auto viterbiToks =
        afToVector<int>(engineObj->criterion->viterbiPath(rawEmission.array()));
    assert(N == viterbiToks.size());

    auto dfalm = DFALM::LM{dfa};

    auto commandDecoder = SimpleDecoder<DFALM::LM, DFALM::State>{
                toW2lDecoderOptions(opts.cmdDecodeOpts),
                dfalm,
                decoderObj->silIdx,
                decoderObj->wordDict.getIndex(kUnkToken),
                transitions};

    DFALM::State commandState;
    commandState.lex = dfalm.dfa; // presumed root state of dfa

    std::vector<int> languageDecode; // stores tokens of a later language decode

    std::string result;

    int i = 0;
    while (i < N) {
        int viterbiSegStart = i;
        while (i < N && viterbiToks[i] == 0)
            ++i;
        int viterbiWordStart = i;
        while (i < N && viterbiToks[i] != 0)
            ++i;
        int viterbiWordEnd = i;
        // it's ok if wordStart == wordEnd, maybe the decoder sees something

        // in the future we could stop the decode after one word instead of
        // decoding everything
        int decodeLen = N - viterbiSegStart;
        commandDecoder.lm_.wordStartsBefore = viterbiWordEnd - viterbiSegStart;
        auto decodeResult = commandDecoder.normal(emissionVec.data() + viterbiSegStart * T, decodeLen, T, commandState);
        auto decoderToks = decodeResult.tokens;
        decoderToks.erase(decoderToks.begin()); // initial hyp token
        std::vector<int> startSil(viterbiSegStart, 0);
        decoderToks.insert(decoderToks.begin(), startSil.begin(), startSil.end());
        decodeLen += viterbiSegStart;

        int j = 0;
        while (j < decodeLen && decoderToks[j] == 0)
            ++j;
        int decodeWordStart = j;
        while (j < decodeLen && decoderToks[j] != 0)
            ++j;
        int decodeWordEnd = j;
        // Again it's ok if wordStart == wordEnd, need to process anyway.
        // Maybe we are in language mode and need to emit those words?

        // we score the decoded word plus any adjacent non-sil viterbi tokens
        // - at least until the next decoded command word start
        int scoreWordStart = std::min(viterbiWordStart, decodeWordStart);
        int scoreWordEnd = decodeWordEnd;
        while (scoreWordEnd < N && viterbiToks[scoreWordEnd] != 0 && decoderToks[scoreWordEnd] == 0)
            ++scoreWordEnd;

        // if the decoder didn't see anything, only discard the area where we allowed
        // words to start.
        if (decodeWordStart == decodeWordEnd)
            scoreWordEnd = viterbiWordEnd;

        i = std::min(scoreWordEnd, N);

        // find the recognized word index
        int outWord = -1;
        // word decode is only written in first silence *after* the word
        for (int j = scoreWordStart; j < std::max(decodeWordEnd + 1, scoreWordEnd); ++j) {
            outWord = decodeResult.words[1 + j - viterbiSegStart]; // +1 because of initial hyp
            if (outWord != -1)
                break;
        }

        // the criterion for rejecting decodes is the decode-score / viterbi-score
        // where the score is the emission-transmission score
        float windowFrac = worstEmissionTransmissionWindowFraction(decoderToks, viterbiToks, scoreWordStart, scoreWordEnd, opts.rejectionTokenWindow);
        bool goodCommand = windowFrac > opts.rejectionThreshold && outWord != -1;

        if (opts.debug) {
            float viterbiScore = emissionTransmissionScore(viterbiToks, scoreWordStart, scoreWordEnd);
            float decoderScore = emissionTransmissionScore(decoderToks, scoreWordStart, scoreWordEnd);

            if (outWord == -1) {
                std::cout << "no command" << std::endl;
            } else if (goodCommand) {
                std::cout << "good command" << std::endl;
            } else {
                std::cout << "rejected command" << std::endl;
            }
            std::cout << "decoder: " << tokensToString(decoderToks, scoreWordStart, scoreWordEnd) << std::endl
                      << "viterbi: " << tokensToString(viterbiToks, scoreWordStart, scoreWordEnd) << std::endl
                      << "scores: decoder: " << decoderScore << " viterbi: " << viterbiScore << " fraction: " << decoderScore / viterbiScore << std::endl
                      << "        worst window fraction: " << windowFrac << std::endl
                      << std::endl;
        }

        // While in language mode, emit language words up to the command
        if (!languageDecode.empty()) {
            for (int j = viterbiSegStart; j < (goodCommand ? decodeWordStart : i); ++j) {
                if (languageDecode[j] != -1) {
                    if (!result.empty())
                        result += " ";
                    result += decoderObj->wordDict.getEntry(languageDecode[j]);
                    if (opts.debug) {
                        std::cout << "decoded language, new out: " << result << std::endl;
                    }
                }
            }
        }

        // If a command was found, output it.
        if (goodCommand) {
            if (!result.empty())
                result += " ";
            result += "@"; // command marker
            result += tokensToStringDedup(decoderToks, decodeWordStart, decodeWordEnd);
            if (opts.debug) {
                std::cout << "decoded command, new out: " << result << std::endl;
            }
        } else {
            if (languageDecode.empty()) {
                // found a non-command and language mode is off
                return nullptr;
            }
            // not a command but we can report whatever language we can find
            continue;
        }

        // if we were decoding multiple commands in one go, the decoder would
        // update the state itself. Since we only do one command at a time,
        // we need to manually update the next start state.
        commandState.lex = dfalm.get(dfalm.dfa, outWord);

        const auto flags = commandState.lex->flags;
        if (flags & DFALM::FLAG_LM) {
            // do a language decode and store the results
            KenFlatTrieLM::State langStartState;
            langStartState.kenState = decoderObj->lm->start(0);
            langStartState.lex = decoderObj->flatTrie->getRoot();
            auto languageResult = decoderObj->decoder->normal(emissionVec.data() + i * T, N - i, T, langStartState);
            languageDecode = std::vector<int>(i, 0);
            languageDecode.insert(languageDecode.end(), languageResult.words.begin() + 1, languageResult.words.end());
            //std::cout << "lang decode: " << tokensToString(languageResult.tokens, 1, languageResult.tokens.size() - 1) << std::endl;
        } else if (flags & DFALM::FLAG_TERM) {
            // TODO: Should reject everything if non-silence follows?
            break;
        } else {
            languageDecode.clear();
        }
    }
    if (!result.empty())
        return strdup(result.c_str());
    return nullptr;
}

} // extern "C"
