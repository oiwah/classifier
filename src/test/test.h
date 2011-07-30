#include <fstream>
#include <sstream>

#include <tool/feature.h>

namespace classifier {
bool ParseFile(const char* file_path,
               std::vector<classifier::datum>* data,
               feature2id* f2i,
               bool libsvm=false) {
  std::vector<classifier::datum>(0).swap(*data);

  std::ifstream ifs(file_path);
  if (!ifs) {
    std::cerr << "cannot open " << file_path << std::endl;
    return false;
  }

  size_t lineN = 0;
  for (std::string line; getline(ifs, line); ++lineN) {
    datum datum;
    std::istringstream iss(line);

    std::string category = "Not defined";
    if (!(iss >> category)) {
      std::cerr << "parse error: you must set category in line " << lineN << std::endl;
      return false;
    }
    datum.category = category;

    if (!libsvm) {
      std::string word = "";
      while (iss >> word) {
        size_t word_id = 0;
        if (f2i->find(word) == f2i->end()) {
          word_id = f2i->size();
          f2i->insert(std::make_pair(word, word_id));
        } else {
          word_id = f2i->at(word);
        }

        datum.fv.push_back(std::make_pair(word_id, 1.0));
      }
    } else {
      size_t id = 0;
      char comma = 0;
      double value = 0.0;
      while (iss >> id >> comma >> value) {
        datum.fv.push_back(std::make_pair(id, value));
      }
    }
    data->push_back(datum);
  }

  return true;
}

template <class T>
void PrintFeatureWeights(T& classifier,
                         const feature_vector& fv) {
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it) {
    size_t word = it->first;
    std::cout << word << std::endl;

    std::vector<std::pair<std::string, double> > results(0);
    classifier.GetFeatureWeight(word, &results);
    for (std::vector<std::pair<std::string, double> >::const_iterator it = results.begin();
         it != results.end();
         ++it) {
      std::cout << it->first << "\t" << it->second << std::endl;
    }
  }
}

template <class T>
int Run (T& classifier,
         const char* classifier_name,
         const std::vector<classifier::datum>& train,
         const std::vector<classifier::datum>& test) {
  std::cout << classifier_name << std::endl;
  classifier.Train(train);

  //PrintFeatureWeights(classifier, train[0].fv);
  size_t score = 0;
  for (size_t i = 0; i < test.size(); ++i) {
    std::string result;
    classifier.Test(test[i].fv, &result);
    if (test[i].category == result)
      ++score;
  }

  std::cout << "accuracy : " << score << " / " << test.size() << std::endl;
  std::cout << std::endl;
  return 0;
}

} //namespace
