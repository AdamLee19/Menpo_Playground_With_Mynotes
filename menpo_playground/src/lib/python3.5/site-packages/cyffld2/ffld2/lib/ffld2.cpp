#include "ffld2.h"

using namespace FFLD;


bool initializePatchWork(const std::vector<InMemoryScene> positive_scenes,
                         const std::vector<InMemoryScene> negative_scenes,
                         const int padx, const int pady,
                         const bool cacheWisdom);


void detect(const Mixture & mixture, const unsigned char* image_array,
            const int width, const int height, const int n_channels,
            const int padding, const int interval, const double threshold,
            const bool cacheWisdom,
            const double overlap, std::vector<Detection> & detections) {

    JPEGImage image(width, height, n_channels, image_array);
    HOGPyramid pyramid(image, padding, padding, interval);

    // Invalid image
    if (pyramid.empty()) {
        return;
    }

    // Couldn't initialize FFTW
    if (!Patchwork::InitFFTW((pyramid.levels()[0].rows() - padding + 15) & ~15,
                             (pyramid.levels()[0].cols() - padding + 15) & ~15,
                             cacheWisdom)) {
        return;
    }

    mixture.cacheFilters();

    // Compute the scores
    std::vector<HOGPyramid::Matrix> scores;
    std::vector<Mixture::Indices> argmaxes;
    std::vector<std::vector<std::vector<Model::Positions> > > positions;

    mixture.convolve(pyramid, scores, argmaxes, &positions);

    // Cache the size of the models
    std::vector<std::pair<int, int> > sizes(mixture.models().size());

    for (unsigned int i = 0; i < sizes.size(); ++i) {
        sizes[i] = mixture.models()[i].rootSize();
    }

    // For each scale
    for (unsigned int z = 0; z < scores.size(); ++z) {
        const double scale = pow(2.0, static_cast<double>(z) / pyramid.interval() + 2);

        const int rows = static_cast<int>(scores[z].rows());
        const int cols = static_cast<int>(scores[z].cols());

        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const double score = scores[z](y, x);

                if (score > threshold) {
                    // Non-maxima suppresion in a 3x3 neighborhood
                    if (((y == 0) || (x == 0) || (score >= scores[z](y - 1, x - 1))) &&
                        ((y == 0) || (score >= scores[z](y - 1, x))) &&
                        ((y == 0) || (x == cols - 1) || (score >= scores[z](y - 1, x + 1))) &&
                        ((x == 0) || (score >= scores[z](y, x - 1))) &&
                        ((x == cols - 1) || (score >= scores[z](y, x + 1))) &&
                        ((y == rows - 1) || (x == 0) || (score >= scores[z](y + 1, x - 1))) &&
                        ((y == rows - 1) || (score >= scores[z](y + 1, x))) &&
                        ((y == rows - 1) || (x == cols - 1) ||
                         (score >= scores[z](y + 1, x + 1)))) {
                        Rectangle bndbox((x - pyramid.padx()) * scale + 0.5,
                                         (y - pyramid.pady()) * scale + 0.5,
                                         sizes[argmaxes[z](y, x)].second * scale + 0.5,
                                         sizes[argmaxes[z](y, x)].first * scale + 0.5);

                        // Truncate the object
                        bndbox.setX(std::max(bndbox.x(), 0));
                        bndbox.setY(std::max(bndbox.y(), 0));
                        bndbox.setWidth(std::min(bndbox.width(), width - bndbox.x()));
                        bndbox.setHeight(std::min(bndbox.height(), height - bndbox.y()));

                        if (!bndbox.empty()) {
                            detections.push_back(Detection(score, bndbox));
                        }
                    }
                }
            }
        }
    }

    // Non maxima suppression
    sort(detections.begin(), detections.end());

    for (unsigned int i = 1; i < detections.size(); ++i)
        detections.resize(remove_if(detections.begin() + i, detections.end(),
                                    Intersector(detections[i - 1], overlap, true)) -
                          detections.begin());
}

void detect(const std::string mixture_filepath,
            const unsigned char* image_array,
            const int width, const int height, const int n_channels,
            const int padding, const int interval, const double threshold,
            const bool cacheWisdom, const double overlap, 
            std::vector<Detection> & detections) {

    // Failed to load mixture model
    Mixture mixture;
    if(!load_mixture_model(mixture_filepath, mixture))
        return;

    detect(mixture, image_array, width, height, n_channels, padding, interval,
           threshold, cacheWisdom, overlap, detections);
}

void detect(const char* mixture_data,
            const unsigned char* image_array,
            const int width, const int height, const int n_channels,
            const int padding, const int interval, const double threshold, 
            const bool cacheWisdom, const double overlap, 
            std::vector<Detection> & detections) {

    // Failed to load mixture model
    Mixture mixture;
    if(!load_mixture_model(mixture_data, mixture))
        return;

    detect(mixture, image_array, width, height, n_channels, padding, interval,
           threshold, cacheWisdom, overlap, detections);
}

bool load_mixture_model(const std::string filepath,
                       Mixture& mixture) {
    std::ifstream in(filepath.c_str(), std::ios::binary);

    if (!in.is_open()) {
        return false;
    }

    in >> mixture;

    return true;
}

bool save_mixture_model(const std::string filepath,
                        const Mixture& mixture) {
    std::ofstream out(filepath.c_str(), std::ios::binary);

    if (!out.is_open()) {
        return false;
    }

    out << mixture;

    return true;
}

bool initializePatchWork(const std::vector<InMemoryScene> positive_scenes,
                         const std::vector<InMemoryScene> negative_scenes,
                         const int padx, const int pady, 
                         const bool cacheWisdom)
{
	int maxRows = 0;
	int maxCols = 0;

	for(unsigned int i = 0; i < positive_scenes.size(); i++) {
	    Scene scene = positive_scenes[i];
        maxRows = std::max(maxRows, (scene.height() + 3) / 4 + pady);
        maxCols = std::max(maxCols, (scene.width() + 3) / 4 + padx);
	}
    for(unsigned int i = 0; i < negative_scenes.size(); i++) {
        Scene scene = negative_scenes[i];
        maxRows = std::max(maxRows, (scene.height() + 3) / 4 + pady);
        maxCols = std::max(maxCols, (scene.width() + 3) / 4 + padx);
    }
    // I'm not sure what this was for, since it doesn't seem to be used
    // in train.cpp
    //nbNegativeScenes -= negative_scenes.size();

	// Initialize the Patchwork class
	if (!Patchwork::InitFFTW((maxRows + 15) & ~15, (maxCols + 15) & ~15, cacheWisdom)) {
		return false;
	}
	return true;
}

bool train(const std::vector<InMemoryScene> positive_scenes,
           const std::vector<InMemoryScene> negative_scenes,
           const int nbComponents,
           const int padx, const int pady, const bool cacheWisdom,
           const int interval, const int nbRelabel,
           const int nbDatamine, const int maxNegatives,
           const double C, const double J,
           const double overlap, const std::string model_out_path)
{
    // Can't build a model from empty scenes
    if (positive_scenes.empty() || negative_scenes.empty()) {
        return false;
    }

    // Initialize FFTW
    if (!initializePatchWork(positive_scenes, negative_scenes, padx, pady, cacheWisdom))
        return false;

	// The mixture to train
	Mixture mixture(nbComponents, positive_scenes);

	if (mixture.empty()) {
	    return false;
	}

    // Pre-train the model
	mixture.trainInMemory(positive_scenes, negative_scenes, padx, pady,
	                      interval, nbRelabel / 2.0, nbDatamine, maxNegatives,
	                      C, J, overlap);

	if (mixture.models()[0].parts().size() == 1)
		mixture.initializeParts(8, std::make_pair(6, 6));

	mixture.trainInMemory(positive_scenes, negative_scenes, padx, pady,
	                      interval, nbRelabel, nbDatamine, maxNegatives, C, J,
				          overlap);

	// Write result to file
	if (!save_mixture_model(model_out_path, mixture))
	    return false;

	return true;
}

bool train(const std::vector<InMemoryScene> positive_scenes,
           const std::vector<InMemoryScene> negative_scenes,
           const int padx, const int pady, const bool cacheWisdom,
           const int interval, const int nbRelabel,
           const int nbDatamine, const int maxNegatives,
           const double C, const double J,
           const double overlap, Mixture& mixture)
{
    // Can't build a model from empty scenes
	if (positive_scenes.empty() || negative_scenes.empty()) {
		return false;
	}

    // Initialize FFTW
    if (!initializePatchWork(positive_scenes, negative_scenes, padx, pady, cacheWisdom))
        return false;

    // If we've been given an empty model, then pre-train first
    mixture.trainInMemory(positive_scenes, negative_scenes, padx, pady,
                          interval, nbRelabel / 2.0, nbDatamine,
                          maxNegatives, C, J, overlap);

	if (mixture.models()[0].parts().size() == 1)
		mixture.initializeParts(8, std::make_pair(6, 6));

	mixture.trainInMemory(positive_scenes, negative_scenes, padx, pady,
	                      interval, nbRelabel, nbDatamine, maxNegatives, C, J,
                              overlap);
	return true;
}
