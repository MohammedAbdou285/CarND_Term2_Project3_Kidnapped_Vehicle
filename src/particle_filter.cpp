/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// use the default random engine generator as it will be used many times
static default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Number of Particles
	num_particles = 500;
	
	// Get the standard deviations and Create Normal Distribution for x, y, theta with nean and std_dev
	double std_dev_x 	= std[0];
	double std_dev_y 	= std[1];
	double std_dev_th 	= std[2];

	normal_distribution<double> dist_x  (x, std_dev_x);
	normal_distribution<double> dist_y  (y, std_dev_y);
	normal_distribution<double> dist_th (theta, std_dev_th);

	// Create particles and their weights based on the num_particles
	particles.resize(num_particles);
	weights.resize(num_particles);

	// Initilaization
	for (unsigned i = 0; i < num_particles; i++)
	{
		particles[i].id 	= i;
		particles[i].x 		= dist_x(gen);
		particles[i].y 		= dist_y(gen);
		particles[i].theta 	= dist_th(gen);
		particles[i].weight = 1.0;
	}

	// CHange the is_initialized to be true
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// In this function, we will implement the prediction which has two cases: 1) yaw_rate = 0 "Special Case", 2) yaw_rate != 0

	// Get the standard deviations
	double std_dev_x  = std_pos[0];
	double std_dev_y  = std_pos[1];
	double std_dev_th = std_pos[2];

	// Create Normal distributions
	normal_distribution<double> dist_x  (0.0, std_dev_x);
	normal_distribution<double> dist_y  (0.0, std_dev_y);
	normal_distribution<double> dist_th (0.0, std_dev_th);
	
	for (unsigned i = 0; i < particles.size(); i++)
	{
		// Special Case
		if (fabs(yaw_rate) <= 0.0001)
		{
			double theta = particles[i].theta;

			particles[i].x 		+= velocity * delta_t * cos(theta) + dist_x(gen);
			particles[i].y 		+= velocity * delta_t * sin(theta) + dist_y(gen);
			particles[i].theta 	+= dist_th(gen);
		}

		// General Case
		else if (fabs(yaw_rate) > 0.0001)
		{
			double theta = particles[i].theta + yaw_rate * delta_t;

			particles[i].x 		+= (velocity / yaw_rate) * ( sin(theta) - sin(particles[i].theta)) + dist_x(gen);
			particles[i].y 		+= (velocity / yaw_rate) * (-cos(theta) + cos(particles[i].theta)) + dist_y(gen);
			particles[i].theta 	 = theta + dist_th(gen);
		}
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned i = 0; i < observations.size(); i++)
	{
		// Set the minimum dist to be the largest possible value 
		double min_dist = numeric_limits<double>::max();

		// Set the ID to be -1
		int landmark_id = -1; 

		for (unsigned j = 0; j < predicted.size(); j++)
		{
			double curr_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (curr_dist < min_dist)
			{
				min_dist = curr_dist;
				landmark_id = predicted[j].id;
			}
		}

		observations[i].id = landmark_id; 
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (unsigned i = 0; i < particles.size(); i++)
	{
		// Define the predictions to filter map landmarks within the sensor range (ROI)
		vector<LandmarkObs> predictions;

		for (unsigned j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			double curr_dist = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);

			if (curr_dist <= sensor_range)
			{
				predictions.push_back(LandmarkObs{	map_landmarks.landmark_list[j].id_i,
													map_landmarks.landmark_list[j].x_f,
													map_landmarks.landmark_list[j].y_f});
			}
		}

		// Transform observations from the vehicle reference to the Map reference
		vector<LandmarkObs> t_observations;

		for (unsigned j = 0; j < observations.size(); j++)
		{
			double trans_x = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
			double trans_y = particles[i].y + sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y;
			t_observations.push_back(LandmarkObs{ observations[j].id, trans_x, trans_y });
		}

		// Do data Associations between the predictions and the transformed observtions
		dataAssociation(predictions, t_observations);

		// Do the Update for the particles based on the multivariate equation 
		for (unsigned j = 0; j < t_observations.size(); j++)
		{
			LandmarkObs trans_obs = t_observations[j];
			LandmarkObs pred;

			int pred_id = trans_obs.id;

			for (unsigned k = 0; k < predictions.size(); k++)
			{
				if (predictions[k].id == pred_id)
				{
					pred.x = predictions[k].x;
					pred.y = predictions[k].y;
				}

			}

			double std_x = std_landmark[0];
			double std_y = std_landmark[1];

			double x_component = -pow(trans_obs.x - pred.x, 2) / (2 * std_x * std_x);
			double y_component = -pow(trans_obs.y - pred.y, 2) / (2 * std_y * std_y); 
			particles[i].weight = (1/(2*M_PI * std_x * std_y)) * exp(x_component + y_component); 
		}
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled_particles;

	double max_weight = 0.0;

	for (unsigned i = 0; i < num_particles; i++)
	{
		weights[i] = particles[i].weight;
		if (weights[i] > max_weight)
		{
			max_weight = weights[i];
		}
	}

	uniform_real_distribution<double> unirealdist(0.0, 2.0 * max_weight);
	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	auto index = uniintdist(gen);


	double beta = 0.0;
	for (unsigned i = 0; i < num_particles; i++)
	{
		beta += unirealdist(gen); //random.random() * 2.0 * Max_w
    	while (beta > weights[index])
		{
			beta -= weights[index];
        	index = (index + 1) % num_particles;
		}
		resampled_particles.push_back(particles[index]); 
	}

    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

	
	// Clear previous associations
	particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
