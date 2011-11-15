#include <sstream>

#include <tool/feature.h>

namespace classifier {
namespace parser {
void NeutralParser(std::istringstream* iss,
                   feature2id* f2i,
                   classifier::datum* datum) {
  std::string word = "";
  while (*iss >> word) {
    size_t word_id = 0;
    if (f2i->find(word) == f2i->end()) {
      word_id = f2i->size();
      f2i->insert(std::make_pair(word, word_id));
    } else {
      word_id = f2i->at(word);
    }

    datum->fv.push_back(std::make_pair(word_id, 1.0));
  }
}
} //namespace parser
} //namespace classifier
