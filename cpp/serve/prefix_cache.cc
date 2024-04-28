/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/prefix_cache.cc
 */
#include "prefix_cache.h"

#include <tvm/runtime/registry.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

// PrefixCacheObj
std::pair<int64_t, int32_t> PrefixCacheObj::InsertSequence(int64_t seq_id, IntTuple tokens) {
  auto [matched_offset, matched_seqs] = radix_tree->MatchPrefix(tokens);
  if (matched_seqs.empty()) {
    radix_tree->AddSequence(seq_id);
    return std::make_pair(0, 0);
  }
  if (matched_offset == tokens.size()) --matched_offset;
  radix_tree->ForkSequence(seq_id, *matched_seqs.begin(), matched_offset);
  return std::make_pair(*matched_seqs.begin(), matched_offset);
}

void PrefixCacheObj::ExtendSequence(int64_t seq_id, IntTuple tokens) {
  while (radix_tree->FreeCapacity() < tokens.size()) {
    CHECK(!recycling_seqs.empty());
    int64_t id = recycling_seqs.begin()->first;
    recycling_seqs[id]();
    recycling_seqs.erase(id);
    radix_tree->RemoveSequence(id);
  }
  radix_tree->ExtendSequence(seq_id, tokens);
}

void PrefixCacheObj::RecycleSequence(int64_t seq_id, PackedFunc callback) {
  CHECK(recycling_seqs.find(seq_id) == recycling_seqs.end());
  recycling_seqs[seq_id] = callback;
}

bool PrefixCacheObj::HasSequence(int64_t seq_id) { return radix_tree->HasSequence(seq_id); }

void PrefixCacheObj::Reset() {
  radix_tree->Reset();
  recycling_seqs.clear();
}

PrefixCache::PrefixCache(size_t num_pages, size_t page_size, size_t num_seqs) {
  ObjectPtr<PrefixCacheObj> n = make_object<PrefixCacheObj>();
  n->radix_tree = PagedRadixTree(num_pages, page_size, num_seqs);
  n->recycling_seqs.clear();
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
