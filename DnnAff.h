#ifndef DNNAFF_H
#define DNNAFF_H

#include "ibex.h"
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>

using namespace ibex;

// Function prototypes
std::vector<Affine2Vector> neural_network_hidden(const std::vector<std::vector<std::vector<double>>>& weights, const std::vector<std::vector<double>>& offsets, const std::vector<Affine2Vector>& input);

std::vector<std::vector<std::vector<double>>> readWeights(const std::string& filename);
std::vector<std::vector<double>> readBiases(const std::string& filename);
std::vector<std::vector<Affine2Vector>> Affine2Splitter(const std::vector<Affine2Vector>& inputs,const int& n_pavage); //pavage de l'entrée
std::vector<Affine2Vector> DeepNeuralNetwork_aff(const std::vector<Affine2Vector>& inputs, const std::string& weightsFile, const std::string& biasesFile);
std::vector<std::vector<Affine2Vector>> Affine2Bisec(const std::vector<Affine2Vector>& inputs);

#endif // DNNAFF_H

