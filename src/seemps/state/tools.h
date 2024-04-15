#pragma once
#include <iterator>
#include "core.h"
#include <pybind11/complex.h>

namespace pybind11 {

class python_list_iterator {
  list list_;
  size_t index_;
  using iterator = python_list_iterator;

public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = object;
  using difference_type = std::ptrdiff_t;
  using pointer = object *;

  class reference {
    const list &list_;
    size_t index_;

  public:
    reference(const list &list, size_t index) : list_{list}, index_{index} {}
    auto operator=(object p) const { return list_[index_] = p; }
    auto operator=(const reference &r) const {
      return list_[index_] = r.list_[r.index_];
    }
    operator object() const { return list_[index_]; }
  };

  iterator(const iterator &) = default;
  iterator(iterator &&) = default;
  iterator &operator=(const iterator &) = default;
  iterator &operator=(iterator &&) = default;
  iterator(list &list, size_t index) : list_{list}, index_{index} {}
  ~python_list_iterator() = default;

  bool operator==(const iterator &other) const {
    return index_ == other.index_;
  }

  bool operator!=(const iterator &other) const {
    return index_ != other.index_;
  }

  iterator &operator++() {
    ++index_;
    return *this;
  }

  iterator operator++(int) {
    auto retval = *this;
    ++index_;
    return retval;
  }

  reference operator*() const { return reference(list_, index_); }
};

class python_list_const_iterator {
  list list_;
  size_t index_;
  using iterator = python_list_const_iterator;

public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = object;
  using difference_type = std::ptrdiff_t;
  using pointer = object *;
  using reference = object &;

  iterator(const iterator &) = default;
  iterator(iterator &&) = default;
  iterator &operator=(const iterator &it) {
    list_ = it.list_;
    index_ = it.index_;
    return *this;
  }
  iterator &operator=(iterator &&it) {
    list_ = std::move(it.list_);
    index_ = it.index_;
    return *this;
  }
  iterator(const list &list, size_t index) : list_{list}, index_{index} {}
  ~python_list_const_iterator() = default;

  bool operator==(const iterator &other) const {
    return index_ == other.index_;
  }

  bool operator!=(const iterator &other) const {
    return index_ != other.index_;
  }

  iterator &operator++() {
    ++index_;
    return *this;
  }

  iterator operator++(int) {
    auto return_value = *this;
    ++index_;
    return return_value;
  }

  auto operator*() const { return list_[index_]; }
};

inline auto begin(list &l) { return python_list_iterator(l, 0); }
inline auto end(list &l) { return python_list_iterator(l, l.size()); }

inline auto begin(const list &l) { return python_list_const_iterator(l, 0); }
inline auto end(const list &l) {
  return python_list_const_iterator(l, l.size());
}

inline auto begin_const(list &l) { return python_list_const_iterator(l, 0); }
inline auto end_const(list &l) {
  return python_list_const_iterator(l, l.size());
}

list copy(const list &l);

object conj(const object &w);

object real(const object &w);

double abs(const object &w);

} // namespace pybind11
