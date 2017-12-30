#include "../Intersector.h"
#include "../Scene.h"
#include "../Mixture.h"

#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>


struct Detection : public FFLD::Rectangle
{
	FFLD::HOGPyramid::Scalar score;

	Detection() : score(0)
	{
	}

	Detection(FFLD::HOGPyramid::Scalar score, FFLD::Rectangle bndbox) : FFLD::Rectangle(bndbox), score(score)
	{
	}

	bool operator<(const Detection & detection) const
	{
		return score > detection.score;
	}
};

bool load_mixture_model(const std::string filepath, FFLD::Mixture& mixture);
bool save_mixture_model(const std::string filepath, const FFLD::Mixture& mixture);

// Detect using a loaded model
void detect(const FFLD::Mixture & mixture, const unsigned char* image,
            const int width, const int height, const int n_channels, const int padding, 
            const int interval, const double threshold, const bool cacheWisdom,
            const double overlap, std::vector<Detection>& detections);
// Detect by loading model from filepath
void detect(const std::string mixture_filepath, const unsigned char* image,
            const int width, const int height, const int n_channels, 
            const int padding, const int interval, const double threshold, 
            const bool cacheWisdom, const double overlap, 
            std::vector<Detection>& detections);

// Train using existing positive and negative scenes
// Suggested parameter values taken from train.cpp:
//     padx = 6, pady = 6, interval = 5, nbRelabel = 5, nbDatamine = 10,
//     maxNegatives = 24000, C = 0.002, J = 2.0, overlap = 0.5,
//     model_out_path = "model.txt"
bool train(const std::vector<FFLD::InMemoryScene> positive_scenes,
           const std::vector<FFLD::InMemoryScene> negative_scenes,
           const int nbComponents,
           const int padx, const int pady, const bool cacheWisdom,
           const int interval, const int nbRelabel,
           const int nbDatamine, const int maxNegatives,
           const double C, const double J,
           const double overlap, const std::string model_out_path);

// Suggested to create Mixture as in training function above:
//      Mixture mixture(nbComponents, positive_scenes);
bool train(const std::vector<FFLD::InMemoryScene> positive_scenes,
           const std::vector<FFLD::InMemoryScene> negative_scenes,
           const int padx, const int pady, const bool cacheWisdom,
           const int interval, const int nbRelabel,
           const int nbDatamine, const int maxNegatives,
           const double C, const double J,
           const double overlap, FFLD::Mixture& mixture);

