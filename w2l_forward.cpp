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
#include "w2l_p.h"

using namespace w2l;

Emission::Emission(EngineBase *engine, af::array emission, af::array inputs) {
    this->engine = engine;
    this->emission = emission;
    this->inputs = inputs;
}

char *Emission::text() {
    auto tokenPrediction =
        afToVector<int>(engine->criterion->viterbiPath(emission));
    auto letters = tknPrediction2Ltr(tokenPrediction, engine->tokenDict);
    if (letters.size() > 0) {
        std::ostringstream ss;
        for (auto s : letters) ss << s;
        return strdup(ss.str().c_str());
    }
    return strdup("");
}

Engine::Engine(const char *acousticModelPath, const char *tokensPath) {
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

Emission *Engine::process(float *samples, size_t sample_count) {
    struct W2lLoaderData data = {};
    std::copy(samples, samples + sample_count, std::back_inserter(data.input));

    auto feat = featurize({data}, {});
    auto input = af::array(feat.inputDims, feat.input.data());
    auto rawEmission = network->forward({fl::input(input)}).front();
    return new Emission(this, rawEmission.array(), input);
}

af::array Engine::process(const af::array &features)
{
    return network->forward({fl::input(features)}).front().array();
}

bool Engine::exportModel(const char *path) {
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

std::vector<float> Engine::transitions() const
{
    return afToVector<float>(criterion->param(0).array());
}

af::array Engine::viterbiPath(const af::array &data) const
{
    return criterion->viterbiPath(data);
}

std::tuple<int, int> Engine::splitOn(std::string s, std::string on) {
    auto split = s.find(on);
    auto first = s.substr(0, split);
    auto second = s.substr(split + on.size());
    // std::cout << "string [" << s << "] on [" << on << "] first " << first << " second " << second << std::endl;
    return {std::stoi(first), std::stoi(second)};
}

std::string Engine::findParens(std::string s) {
    auto start = s.find('(');
    auto end = s.find(')', start);
    auto sp = s.substr(start + 1, end - start - 1);
    // std::cout << "string split [" << s << "] " << start << " " << end << " [" << sp << "]" << std::endl;
    return sp;
}

void Engine::exportParams(std::ofstream& f, fl::Variable params) {
    auto array = afToVector<float>(params.array());
    for (auto& p : array) {
        f << std::hex << (uint32_t&)p;
        if (&p != &array.back()) {
            f << " ";
        }
    }
    f << std::dec;
}

bool Engine::exportLayer(std::ofstream& f, fl::Module *module) {
    auto pretty = module->prettyString();
    auto type = pretty.substr(0, pretty.find(' '));
    std::cout << "[w2lapi] exporting: " << pretty << std::endl;
    if (type == "WeightNorm") {
        auto wn = dynamic_cast<fl::WeightNorm *>(module);
        auto lastParam = pretty.rfind(",") + 2;
        auto dim = pretty.substr(lastParam, pretty.size() - lastParam - 1);
        f << "WN " << dim << " ";
        exportLayer(f, wn->module().get());
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

