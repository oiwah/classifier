#include <fstream>
#include <sstream>

#include <tool/feature.h>
#include <parser/neutral.h>
#include <parser/libsvm.h>

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

    if (!libsvm)
      parser::NeutralParser(&iss, f2i, &datum);
    else
      parser::LibsvmParser(&iss, &datum);
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
