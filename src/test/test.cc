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

namespace {
} //namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " [training file] [test file]" << std::endl;
    return -1;
  }

  classifier::feature2id f2i(0);
  std::vector<classifier::datum> train;
  if (!ParseFile(argv[1], &train, &f2i, true))
    return -1;

  std::vector<classifier::datum> test;
  if (!ParseFile(argv[2], &test, &f2i, true))
    return -1;
  /**
  classifier::naivebayes::NaiveBayes nb;
  nb.set_alpha(1.5);
  if (classifier::Run(nb, "NaiveBayes", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::naivebayes::ComplementNaiveBayes c_nb;
  c_nb.set_alpha(1.5);
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

  classifier::pa::PA pa_one(1);
  pa_one.SetC(1.0);
  if (classifier::Run(pa_one, "PassiveAggressive-I", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::pa::PA pa_two(2);
  pa_two.SetC(1.0);
  if (classifier::Run(pa_two, "PassiveAggressive-II", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }
  */
  classifier::cw::CW cw(0.001);
  if (classifier::Run(cw, "ConfidenceWeighted", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::cw::CW scw1(0.001);
  scw1.ChangeMode(1);
  scw1.SetC(1.0);
  if (classifier::Run(scw1, "SoftConfidenceWeighted-I", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }  

  classifier::cw::CW scw2(0.001);
  scw2.ChangeMode(2);
  scw2.SetC(1.0);
  if (classifier::Run(scw2, "SoftConfidenceWeighted-II", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }  
  /*
  classifier::arow::AROW arow(0.01);
  if (classifier::Run(arow, "AROW", train, test) == -1) {
    std::cerr << "AROW failed." << std::endl;
  }

  classifier::subgradient::SubgradientHinge sgh;
  if (classifier::Run(sgh, "SubgradientHinge", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::subgradient::ASGDHinge asgdh;
  if (classifier::Run(asgdh, "ASGDHinge", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::fobos::FOBOS fobos(1.0, 0.001);
  if (classifier::Run(fobos, "FOBOS", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::fobos::CumulativeFOBOS cfobos(1.0, 0.001);
  if (classifier::Run(cfobos, "CumulativeFOBOS", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::dual_averaging::DualAveraging da(0.001);
  if (classifier::Run(da, "DualAveraging", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  classifier::loglinear::LogLinearSGD llsgd;
  if (classifier::Run(llsgd, "LogLinearSGD", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }
  **/
  return 0;
}
