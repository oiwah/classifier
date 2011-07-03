#include <confidence_weighted/cw.h>

#include <fstream>
#include <sstream>

namespace {
bool ParseFile(const char* file_path,
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
    if (!(iss >> category)) {
      std::cerr << "parse error: you must set category in line " << lineN << std::endl;
      return false;
    }
    datum.category = category;

    std::vector<std::string> words(0);
    std::string word;
    while (iss >> word)
      datum.fv[word] += 1.0;

    data->push_back(datum);
  }

  return true;
}

void PrintFeatureScores(const classifier::cw::CW& cw,
                        const std::vector<classifier::datum>& train) {
  for (classifier::feature_vector::const_iterator it = train[0].fv.begin();
       it != train[0].fv.end();
       ++it) {
    const std::string word = it->first;
    std::cout << word << std::endl;

    std::vector<std::pair<std::string, double> > results(0);
    cw.CompareFeatureWeight(word, &results);
    for (std::vector<std::pair<std::string, double> >::const_iterator it = results.begin();
         it != results.end();
         ++it) {
      std::cout << it->first << "\t" << it->second << std::endl;
    }
  }
}
} //namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " [training file] [test file]" << std::endl;
    return -1;
  }

  classifier::cw::CW cw(0.5);

  std::vector<classifier::datum> train;
  if (!ParseFile(argv[1], &train))
    return -1;
  cw.Train(train);

  PrintFeatureScores(cw, train);

  std::vector<classifier::datum> test;
  if (!ParseFile(argv[2], &test))
    return -1;

  size_t score = 0;
  for (size_t i = 0; i < test.size(); ++i) {
    std::string result;
    cw.Test(test[i].fv, &result);
    if (test[i].category == result)
      ++score;
    std::cout << i << "th data : " << test[i].category << "\t" << result << std::endl;
  }

  std::cout << "accuracy : " << score << " / " << test.size() << std::endl;
  return 0;
}
