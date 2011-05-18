#include <naivebayes/nb.h>

#include <fstream>
#include <sstream>

namespace {
bool ParseFile(bool test,
               const char* file_path,
               std::vector<classifier::datum>* data) {
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
    if (!test && !(iss >> category)) {
      std::cerr << "parse error: you must set category in line " << lineN << std::endl;
      return false;
    }
    datum.category = category;

    std::vector<std::string> words(0);
    std::string word;
    while (iss >> word) {
      words.push_back(word);
    }
    datum.words = words;

    data->push_back(datum);
  }

  return true;
}

void PrintFeatureScores(const classifier::naivebayes::NaiveBayes& nb,
                        const std::vector<classifier::datum>& train) {
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      std::vector<std::pair<std::string, double> > results(0);
      nb.CompareFeatureWeight(train[i].words[j], &results);
      std::cout << train[i].words[j] << std::endl;
      for (std::vector<std::pair<std::string, double> >::const_iterator it = results.begin();
           it != results.end();
           ++it) {
        std::cout << it->first << "\t" << it->second << std::endl;
      }
    }
  }
}
} //namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [training file] [test file]" << std::endl;
    return -1;
  }
  
  classifier::naivebayes::NaiveBayes nb;
  nb.set_alpha(1.2);

  std::vector<classifier::datum> train;
  if (!ParseFile(false, argv[1], &train))
    return -1;
  nb.Train(train);

  PrintFeatureScores(nb, train);
  
  std::vector<classifier::datum> test;
  if (!ParseFile(true, argv[2], &test))
    return -1;

  for (size_t i = 0; i < test.size(); ++i) {
    std::string result;
    nb.Test(test[i], &result);
    std::cout << i << "th data : " << result << std::endl;
  }
  
  return 0;
}
