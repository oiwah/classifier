#include <fstream>
#include <sstream>

#include <tool/feature.h>

namespace classifier {
bool ParseFile(const char* file_path,
               std::vector<classifier::datum>* data,
               bool libsvm=false) {
  std::vector<classifier::datum>(0).swap(*data);

  std::ifstream ifs(file_path);
  if (!ifs) {
    std::cerr << "cannot open " << file_path << std::endl;
    return false;
  }

  size_t lineN = 0;
  for (std::string line; getline(ifs, line); ++lineN) {
    classifier::datum datum;
    std::istringstream iss(line);

    std::string category = "Not defined";
    if (!(iss >> category)) {
      std::cerr << "parse error: you must set category in line " << lineN << std::endl;
      return false;
    }
    datum.category = category;

    std::vector<std::string> words(0);

    if (!libsvm) {
      std::string word;
      while (iss >> word)
        datum.fv[word] += 1.0;
    } else {
      size_t id;
      while (iss >> id) {
        char comma;
        iss >> comma;
        
        double value;
        iss >> value;
        
        std::ostringstream oss;
        oss << id;
        std::string word = oss.str();
        
        datum.fv[word] += value;
      }
    }
    data->push_back(datum);
  }

  return true;
}

template <class T>
void PrintFeatureWeights(T& classifier,
                         const classifier::feature_vector fv) {
  for (classifier::feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it) {
    const std::string word = it->first;
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
