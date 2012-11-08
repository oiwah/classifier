#include <naivebayes/nb.h>
#include <complement_nb/complement_nb.h>

#include <perceptron/perceptron.h>
#include <perceptron/averaged_perceptron.h>
#include <passive_aggressive/pa.h>
#include <confidence_weighted/cw.h>
#include <arow/arow.h>

#include <subgradient/hinge.h>
#include <subgradient/averaged_hinge.h>
#include <fobos/fobos.h>
#include <fobos/cumulative_fobos.h>
#include <dual_averaging/da.h>

#include <loglinear/loglinear_sgd.h>

#include <test/test.h>
#include <test/cmdline.h>

namespace {
std::string algo_name = "select algorithm(All/NB/CNB/P/AP/PA/PA1/PA2/CW/SGD/ASGD/FOBOS/CFOBOS/DA/LL)";
enum algo_num {
  All = 0,
  None = 9,

  NaiveBayes = 11,
  ComplementNaiveBayes = 12,
  Perceptron = 21,
  AveragedPerceptron = 22,
  PassiveAggressive = 31,
  PassiveAggressiveI = 32,
  PassiveAggressiveII = 33,
  ConfidenceWeighted = 41,

  SubgradientHinge = 101,
  AveragedSubgradientHinge = 102,
  FOBOS = 111,
  CumulativeFOBOS = 112,
  DualAveraging = 121,

  LogLinearSGD = 201,
};

algo_num SelectAlgo(const std::string& algo_identifier) {
  if (algo_identifier == "All") return All;
  else if (algo_identifier == "NB") return NaiveBayes;
  else if (algo_identifier == "CNB") return ComplementNaiveBayes;
  else if (algo_identifier == "P") return Perceptron;
  else if (algo_identifier == "AP") return AveragedPerceptron;
  else if (algo_identifier == "PA") return PassiveAggressive;
  else if (algo_identifier == "PAI") return PassiveAggressiveI;
  else if (algo_identifier == "PAII") return PassiveAggressiveII;
  else if (algo_identifier == "SGD") return SubgradientHinge;
  else if (algo_identifier == "ASGD") return AveragedSubgradientHinge;
  else if (algo_identifier == "FOBOS") return FOBOS;
  else if (algo_identifier == "CFOBOS") return CumulativeFOBOS;
  else if (algo_identifier == "DA") return DualAveraging;
  else if (algo_identifier == "LL") return LogLinearSGD;
  else return None;
}
} //namespace

int main(int argc, char** argv) {
  cmdline::parser parser;
  parser.add<std::string>("train", 't', "train file path", true);
  parser.add<std::string>("classify", 'c', "classify file path", true);
  parser.add<std::string>("algorithm", 'a', algo_name, true, "All");
  parser.add<double>("alpha", '\0', "alpha (default : 1.5) [NB/CNB]", false, 1.5);
  parser.add<double>("C", '\0', "C (default : 1.0) [PA-I|PA-II]", false, 1.0);

  parser.parse_check(argc, argv);

  classifier::feature2id f2i(0);
  std::vector<classifier::datum> train;
  if (!ParseFile(parser.get<std::string>("train"), &train, &f2i, true))
    return -1;

  std::vector<classifier::datum> test;
  if (!ParseFile(parser.get<std::string>("classify"), &test, &f2i, true))
    return -1;
  algo_num algo = SelectAlgo(parser.get<std::string>("algorithm"));

  if (algo == None) {
    parser.usage();
    return -1;
  }

  RunClassifier(algo);
  return 0;
}
