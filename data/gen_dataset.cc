#include <iostream>
#include <cassert>
#include <algorithm>

namespace {
enum EmotionType {
  EXCELLENT = 0,
  GOOD = 1,
  SO_SO = 2,
  BAD = 3,
  POOR = 4,
  NUM_EMOTION_TYPES = 5,
};

const int kMaxLevel = 100;
const int kPositive[] = {40, 80, 90, 95, 100};
const int kNegative[] = {5, 10, 20, 60, 100};
const int kNeutral[] = {10, 20, 80, 90, 100};

const int kPersonLevel = 100;
const int kPersonDistribution[] = {33, 66};

const int kDataSize = 10000;
const int kEmotionSizePerPerson = 10;

bool IsValidEmotionType(EmotionType emotion_type) {
  return (emotion_type >= 0 && emotion_type < NUM_EMOTION_TYPES);
}

int Random(int size) {
  return static_cast<int> (1.0 * size * rand() / (RAND_MAX + 1.0));
}

int arraysize() {
  return sizeof(kPositive) / sizeof(kPositive[0]);
}

void PrintEmotion(const EmotionType& emotion_type) {
  switch(emotion_type) {
    case EXCELLENT:
      std::cout << "Excellent ";
      break;
    case GOOD:
      std::cout << "Good ";
      break;
    case SO_SO:
      std::cout << "So-so ";
      break;
    case BAD:
      std::cout << "Bad ";
      break;
    case POOR:
      std::cout << "Poor ";
      break;
    default:
      assert(-1);
  }
}

} //namespace


int main() {
  for (int i = 0; i < kDataSize; ++i) {
    const int *levels = kNeutral;
    int person_type = Random(kPersonLevel);

    if (person_type < kPersonDistribution[0]) {
      std::cout << "Positive ";
      levels = kPositive;
    } else if (person_type < kPersonDistribution[1]) {
      std::cout << "Negative ";
      levels = kNegative;
    } else {
      std::cout << "Neutral ";
    }

    for (int j = 0; j < kEmotionSizePerPerson; ++j) {
      EmotionType emotion_type = POOR;
      int level = Random(kMaxLevel);

      for (int k = 0; k < arraysize(); ++k) {
        if (level <= levels[k]) {
          emotion_type = static_cast<EmotionType>(k);
          break;
        }
      }

      PrintEmotion(emotion_type);
    }
    std::cout << std::endl;
  }
}
