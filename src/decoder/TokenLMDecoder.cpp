/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>

#include "decoder/TokenLMDecoder.h"

namespace w2l {

void TokenLMDecoder::mergeCandidates() {
}

void TokenLMDecoder::decodeStep(const float* emissions, int T, int N) {
  int startFrame = nDecodedFrames_ - nPrunedFrames_;
  // Extend hyp_ buffer
  if (hyp_.size() < startFrame + T + 2) {
    for (int i = hyp_.size(); i < startFrame + T + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconDecoderState>());
    }
  }

  // Looping over all the frames
  for (int t = 0; t < T; t++) {
    candidatesReset();
    for (const LexiconDecoderState& prevHyp : hyp_[startFrame + t]) {
      const LMStatePtr& prevLmState = prevHyp.lmState;
      const TrieNode* prevLex = prevHyp.lex;
      const int prevIdx = prevLex->idx;

      /* (1) Try children */
      for (auto& child : prevLex->children) {
        int n = child.first;
        const TrieNode* lex = child.second.get();
        float score = prevHyp.score + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType == CriterionType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silWeight;
        }

        auto lmScoreReturn = lm_->score(prevLmState, n);
        score += lmScoreReturn.second * opt_.lmWeight;

        // We eat-up a new token
        if (opt_.criterionType != CriterionType::CTC || prevHyp.getPrevBlank() ||
            n != prevIdx) {
          if (!lex->children.empty()) {
            candidatesAdd(
                lmScoreReturn.first,
                lex,
                &prevHyp,
                score,
                n,
                -1,
                false // prevBlank
            );
          }
        }

        // If we got a true word
        for (int i = 0; i < lex->nLabel; i++) {
          candidatesAdd(
              lmScoreReturn.first,
              lexicon_->getRoot(),
              &prevHyp,
              score + opt_.wordScore,
              n,
              lex->label[i],
              false // prevBlank
          );
        }

        // If we got an unknown word and we want to emit
        if (lex->nLabel == 0 && (opt_.unkScore > kNegativeInfinity)) {
          candidatesAdd(
              lmScoreReturn.first,
              lexicon_->getRoot(),
              &prevHyp,
              score + opt_.unkScore,
              n,
              unk_,
              false // prevBlank
          );
        }
      }

      /* (2) Try same lexicon node */
      if (opt_.criterionType != CriterionType::CTC || !prevHyp.getPrevBlank()) {
        int n = prevIdx;
        float score = prevHyp.score + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType == CriterionType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silWeight;
        }

        candidatesAdd(
            prevLmState,
            prevLex,
            &prevHyp,
            score,
            n,
            -1,
            false // prevBlank
        );
      }

      /* (3) CTC only, try blank */
      if (opt_.criterionType == CriterionType::CTC) {
        int n = blank_;
        float score = prevHyp.score + emissions[t * N + n];
        candidatesAdd(
            prevLmState,
            prevLex,
            &prevHyp,
            score,
            n,
            -1,
            true // prevBlank
        );
      }
    }

    candidatesStore(hyp_[startFrame + t + 1], false);
    updateLMCache(lm_, hyp_[startFrame + t + 1]);
  }
  nDecodedFrames_ += T;
}

} // namespace w2l
