//--------------------------------------------------------------------------------------------------
// Implementation of the papers "Exact Acceleration of Linear Object Detectors", 12th European
// Conference on Computer Vision, 2012 and "Deformable Part Models with Individual Part Scaling",
// 24th British Machine Vision Conference, 2013.
//
// Copyright (c) 2013 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
//
// This file is part of FFLDv2 (the Fast Fourier Linear Detector version 2)
//
// FFLDv2 is free software: you can redistribute it and/or modify it under the terms of the GNU
// Affero General Public License version 3 as published by the Free Software Foundation.
//
// FFLDv2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
// General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with FFLDv2. If
// not, see <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------

#ifndef FFLD_MIXTURE_H
#define FFLD_MIXTURE_H

#include "Model.h"
#include "Patchwork.h"
#include "Scene.h"

namespace FFLD
{
/// The Mixture class represents a mixture of deformable part-based models.
class Mixture
{
public:
	/// Type of a matrix of indices.
	typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Indices;
	
	/// Constructs an empty mixture. An empty mixture has no model.
	Mixture();
	
	/// Constructs a mixture from parameters.
	/// @param[in] models A list of models (mixture components).
	explicit Mixture(const std::vector<Model> & models);
	
	/// Constructs a mixture with the specified number of mixture components. The sizes of the
	/// models are determined from the sizes of the objects using Felzenszwalb's heuristic.
	/// @param[in] nbComponents Number of mixture components (without symmetry).
	/// @param[in] scenes Scenes to use for training.
	/// @param[in] name Name of the objects to detect.
	template <typename T>
	Mixture(int nbComponents, const std::vector<T> & scenes, Object::Name name) : cached_(false), zero_(true)
    {
        // Create an empty mixture if any of the given parameters is invalid
        if ((nbComponents <= 0) || scenes.empty()) {
            return;
        }

        // Compute the root filters' sizes using Felzenszwalb's heuristic
        const std::vector<std::pair<int, int> > sizes = FilterSizes(nbComponents, scenes, name);
        init_models_with_sizes(sizes, nbComponents);
    }

	/// Constructs a mixture with the specified number of mixture components. The sizes of the
    /// models are determined from the sizes of the objects using Felzenszwalb's heuristic.
    /// @param[in] nbComponents Number of mixture components (without symmetry).
    /// @param[in] positive_scenes Scenes to use for training.
    template <typename T>
    Mixture(int nbComponents, const std::vector<T> & positive_scenes) : cached_(false), zero_(true)
    {
    	// Create an empty mixture if any of the given parameters is invalid
    	if ((nbComponents <= 0) || positive_scenes.empty()) {
    		return;
    	}

    	// Compute the root filters' sizes using Felzenszwalb's heuristic
    	const std::vector<std::pair<int, int> > sizes = FilterSizes(nbComponents, positive_scenes);
        init_models_with_sizes(sizes, nbComponents);
    }

	/// Returns whether the mixture is empty. An empty mixture has no model.
	bool empty() const;
	
	/// Returns the list of models (mixture components).
	const std::vector<Model> & models() const;
	
	/// Returns the list of models (mixture components).
	std::vector<Model> & models();
	
	/// Returns the minimum root filter size (<tt>rows x cols</tt>).
	std::pair<int, int> minSize() const;
	
	/// Returns the maximum root filter size (<tt>rows x cols</tt>).
	std::pair<int, int> maxSize() const;
	
	/// Trains the mixture.
	/// @param[in] scenes Scenes to use for training.
	/// @param[in] name Name of the objects to detect.
	/// @param[in] padx Amount of horizontal zero padding (in cells).
	/// @param[in] pady Amount of vertical zero padding (in cells).
	/// @param[in] interval Number of levels per octave in the pyramid.
	/// @param[in] nbRelabel Number of training iterations.
	/// @param[in] nbDatamine Number of data-mining iterations within each training iteration.
	/// @param[in] maxNegatives Maximum number of hard negative examples to sample.
	/// @param[in] C Regularization constant of the SVM.
	/// @param[in] J Weighting factor of the positives.
	/// @param[in] overlap Minimum overlap in latent positive search.
	/// @returns The final SVM loss.
	/// @note The magic constants come from Felzenszwalb's implementation.
	double train(const std::vector<Scene> & scenes, Object::Name name, int padx = 12, int pady = 12,
				 int interval = 5, int nbRelabel = 5, int nbDatamine = 10, int maxNegatives = 24000,
				 double C = 0.002, double J = 2.0, double overlap = 0.7);

    /// Trains the mixture using in-memory scenes of separate guaranteed positive and negative
    /// examples.
    /// @param[in] positive_scenes Scenes to use for positive training.
    /// @param[in] negative_scenes Scenes to use for negative training.
    /// @param[in] name Name of the objects to detect.
    /// @param[in] padx Amount of horizontal zero padding (in cells).
    /// @param[in] pady Amount of vertical zero padding (in cells).
    /// @param[in] interval Number of levels per octave in the pyramid.
    /// @param[in] nbRelabel Number of training iterations.
    /// @param[in] nbDatamine Number of data-mining iterations within each training iteration.
    /// @param[in] maxNegatives Maximum number of hard negative examples to sample.
    /// @param[in] C Regularization constant of the SVM.
    /// @param[in] J Weighting factor of the positives.
    /// @param[in] overlap Minimum overlap in latent positive search.
    /// @returns The final SVM loss.
    /// @note The magic constants come from Felzenszwalb's implementation.
    double trainInMemory(const std::vector<InMemoryScene> & positive_scenes,
                         const std::vector<InMemoryScene> & negative_scenes,
                         const int padx = 12, const int pady = 12,
                         const int interval = 5, const int nbRelabel = 5,
                         const int nbDatamine = 10, const int maxNegatives = 24000,
                         const double C = 0.002, const double J = 2.0,
                         const double overlap = 0.7);
	
	/// Initializes the specidied number of parts from the root of each model.
	/// @param[in] nbParts Number of parts (without the root).
	/// @param[in] partSize Size of each part (<tt>rows x cols</tt>).
	void initializeParts(int nbParts, std::pair<int, int> partSize);
	
	/// Returns the scores of the convolutions + distance transforms of the models with a
	/// pyramid of features (useful to compute the SVM margins).
	/// @param[in] pyramid Pyramid of features.
	/// @param[out] scores Scores for each pyramid level.
	/// @param[out] argmaxes Indices of the best model (mixture component) for each pyramid
	/// level.
	/// @param[out] positions Positions of each part of each model for each pyramid level
	/// (<tt>models x parts x levels</tt>).
	void convolve(const HOGPyramid & pyramid, std::vector<HOGPyramid::Matrix> & scores,
				  std::vector<Indices> & argmaxes,
				  std::vector<std::vector<std::vector<Model::Positions> > > * positions = 0) const;
	
	/// Caches the transformed version of the models' filters.
	void cacheFilters() const;
	
private:
    void init_models_with_sizes(const std::vector<std::pair<int, int> > sizes,
                                const int nbComponents);
	// Extracts all the positives
	void posLatentSearch(const std::vector<Scene> & scenes, const Object::Name name,
						 const int padx, const int pady, const int interval, const double overlap,
						 std::vector<std::pair<Model, int> > & positives) const;

	void posLatentSearchInMemory(const std::vector<InMemoryScene> & scenes,
                                 const int padx, const int pady, const int interval, const double overlap,
    							 std::vector<std::pair<Model, int> > & positives) const;

    bool scorePositiveScene(const JPEGImage image, const std::vector<Object> objects,
                            const int padx, const int pady, const int interval, double overlap,
                            std::vector<std::pair<Model, int> > & positives) const;
	
	// Bootstraps negatives with a non zero loss
	void negLatentSearch(const std::vector<Scene> & scenes, const Object::Name name,
						 const int padx, const int pady, const int interval, const int maxNegatives,
						 std::vector<std::pair<Model, int> > & negatives) const;

	void negLatentSearchInMemory(const std::vector<InMemoryScene> & scenes,
                                 const int padx, const int pady, const int interval, const int maxNegatives,
    							 std::vector<std::pair<Model, int> > & negatives) const;

    bool scoreNegativeScene(const JPEGImage image, const unsigned int scene_index,
                            const int nbCached, const int padx, const int pady, const int interval, const int maxNegatives,
                            std::vector<std::pair<Model, int> > & negatives, unsigned int& current_count) const;
	
	// Trains the mixture from positive and negative samples with fixed latent variables
	double train(const std::vector<std::pair<Model, int> > & positives,
				 const std::vector<std::pair<Model, int> > & negatives, double C, double J,
				 int maxIterations = 400);
	
	// Returns the scores of the convolutions + distance transforms of the models with a pyramid of
	// features (useful to compute the SVM margins)
	void convolve(const HOGPyramid & pyramid,
				  std::vector<std::vector<HOGPyramid::Matrix> > & scores,
				  std::vector<std::vector<std::vector<Model::Positions> > > * positions = 0) const;
	
	// Computes the size of the roots of the models
	template <typename T>
	std::vector<std::pair<int, int> > FilterSizes(const int nbComponents, const std::vector<T> & scenes)
    {
        std::vector<Object> positive_objects;

    	for (int i = 0; i < scenes.size(); ++i) {
            for (int j = 0; j < scenes[i].objects().size(); ++j) {
                positive_objects.push_back(scenes[i].objects()[j]);
            }
        }

        return FilterSizes(nbComponents, positive_objects);
    }

    template <typename T>
    std::vector<std::pair<int, int> > FilterSizes(const int nbComponents, const std::vector<T> & scenes, Object::Name name)
    {
        std::vector<Object> positive_objects;

    	for (int i = 0; i < scenes.size(); ++i) {
            for (int j = 0; j < scenes[i].objects().size(); ++j) {
                const Object & obj = scenes[i].objects()[j];

                if ((obj.name() == name) && !obj.difficult())
                    positive_objects.push_back(obj);
            }
        }

        return FilterSizes(nbComponents, positive_objects);
    }
    static std::vector<std::pair<int, int> > FilterSizes(const int nbComponents,
                                                         const std::vector<Object> & positive_objects);

	// Attempts to split samples into a left facing cluster and a right facing cluster
	static void Cluster(int nbComponents, std::vector<std::pair<Model, int> > & samples);
	
	std::vector<Model> models_;
	
	mutable std::vector<Patchwork::Filter> filterCache_; // Cache of transformed filters
	mutable bool cached_; // Whether the current filters have been cached
	mutable bool zero_; // Whether the current filters are zero
};

/// Serializes a mixture to a stream.
std::ostream & operator<<(std::ostream & os, const Mixture & mixture);

/// Unserializes a mixture from a stream.
std::istream & operator>>(std::istream & is, Mixture & mixture);
}

#endif
