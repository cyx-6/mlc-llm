/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/prefix_cache.h
 */
#ifndef MLC_LLM_SERVE_PREFIX_CACHE_H_
#define MLC_LLM_SERVE_PREFIX_CACHE_H_
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

#include <unordered_map>
#include <unordered_set>

#include "radix_tree.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

class PrefixCacheObj : public Object {
 public:
  PagedRadixTree radix_tree;
  std::unordered_map<int64_t, PackedFunc> recycling_seqs;

  /*!
   * \brief Insert a new tokenized sequence into Prefix Cache.
   * \param tokens The tokens of tokenized sequence.
   * \return The pair of sequence id and the lenghth of prefix which has been prefilled.
   */
  std::pair<int64_t, int32_t> InsertSequence(int64_t seq_id, IntTuple tokens);

  /*!
   * \brief Extend a sequence with new tokenized sequence suffix.
   * \param seq_id The sequence to be extneded.
   * \param tokens The tokens of tokenized sequence suffix to extend.
   * \throw Error if the given sequence id is not valid or active.
   */
  void ExtendSequence(int64_t seq_id, IntTuple tokens);

  /*!
   * \brief Recycle a sequence. The recycled sequence will not be removed immediately, as long as
   memory is sufficient. And it will be reused again if it is exactly matched in the future request.
   * \param seq_id The sequence to be recycled.
   * \throw Error if the given sequence id is not valid.
   */
  void RecycleSequence(int64_t seq_id, PackedFunc callback);

  bool HasSequence(int64_t seq_id);

  void Reset();

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "mlc.serve.PrefixCache";
  TVM_DECLARE_BASE_OBJECT_INFO(PrefixCacheObj, Object)
};

TVM_REGISTER_OBJECT_TYPE(PrefixCacheObj);

class PrefixCache : public ObjectRef {
 public:
  /*!
   * \brief Constructor of paged radix tree.
   * \param num_pages The number of radix tree pages.
   * \param page_size The page size of each radix tree page.
   * \param num_seqs The maximum number of sequence ID.
   */
  explicit PrefixCache(size_t num_pages, size_t page_size, size_t num_seqs);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PrefixCache, ObjectRef, PrefixCacheObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_PREFIX_CACHE_H_
