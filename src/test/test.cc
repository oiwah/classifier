#include <naivebayes/nb.h>
#include <complement_nb/complement_nb.h>

#include <perceptron/perceptron.h>
#include <perceptron/averaged_perceptron.h>
#include <passive_aggressive/pa.h>
#include <confidence_weighted/cw.h>

#include <subgradient/hinge.h>
#include <fobos/fobos.h>

#include <test/test.h>
//#include <test/libsvm_test.h>

namespace {
} //namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " [training file] [test file]" << std::endl;
    return -1;
  }

  std::vector<classifier::datum> train;
  if (!ParseFile(argv[1], &train))
    return -1;

  std::vector<classifier::datum> test;
  if (!ParseFile(argv[2], &test))
    return -1;

  classifier::naivebayes::NaiveBayes nb;
  if (classifier::Run(nb, "NaiveBayes", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::naivebayes::ComplementNaiveBayes c_nb;
  if (classifier::Run(c_nb, "ComplementNaiveBayes", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::perceptron::Perceptron perc;
  if (classifier::Run(perc, "Perceptron", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::perceptron::AveragedPerceptron av_perc;
  if (classifier::Run(av_perc, "AveragedPerceptron", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::pa::PA pa;
  if (classifier::Run(pa, "PassiveAggressive", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::cw::CW cw(0.1);
  if (classifier::Run(cw, "ConfidenceWeighted", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::subgradient::SubgradientHinge sgh;
  if (classifier::Run(sgh, "SubgradientHinge", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::fobos::FOBOS fobos(1.0, 0.00001);
  if (classifier::Run(fobos, "FOBOS", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }
  
  return 0;
}
