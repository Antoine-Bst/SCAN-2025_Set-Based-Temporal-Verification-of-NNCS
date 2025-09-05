#include "ibex.h"
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include "DnnAff.h"

using namespace ibex;

#define __PREC__ 1e-7
#define __METH__ HEUN
#define __DURATION__ 0.1

int nb_sigmoid = 0;

void printWeightsAndBiases(const std::vector<std::vector<std::vector<double>>>& weights,
                           const std::vector<std::vector<double>>& biases) {
    std::cout << "Loaded Weights and Biases:" << std::endl;

    // Print weights for each layer
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        std::cout << "Layer " << layer + 1 << " Weights:" << std::endl;
        const auto& layerWeights = weights[layer];
        for (const auto& row : layerWeights) {
            for (double weight : row) {
                std::cout << weight << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Print biases for each layer
    for (size_t layer = 0; layer < biases.size(); ++layer) {
        std::cout << "Layer " << layer + 1 << " Biases:" << std::endl;
        for (double bias : biases[layer]) {
            std::cout << bias << " ";
        }
        std::cout << std::endl << std::endl;
    }
}


std::vector<std::vector<std::vector<double>>> readWeights(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<std::vector<double>>> weights; // 3D vector for all layers
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return weights;
    }

    std::string line;
    std::vector<std::vector<double>> current_layer;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue; // Skip empty lines
        }

        if (line.find("Weights") != std::string::npos) {
            // Start a new layer
            if (!current_layer.empty()) {
                weights.push_back(current_layer);
                current_layer.clear();
            }
        } else {
            // Parse a row of weights
            std::istringstream iss(line);
            std::vector<double> row;
            double value;
            while (iss >> value) {
                row.push_back(value);
            }
            current_layer.push_back(row);
        }
    }

    // Add the last layer
    if (!current_layer.empty()) {
        weights.push_back(current_layer);
    }

    file.close();

    // Debugging: Print parsed weights dimensions
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        size_t rows = weights[layer].size();
        size_t cols = (rows > 0 ? weights[layer][0].size() : 0);
        std::cout << "Layer " << layer + 1 << ": " << rows << " x " << cols << std::endl;
    }

    return weights;
}


std::vector<std::vector<double>> readBiases(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> biases;
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return biases;
    }

    std::string line;
    std::vector<double> current_layer;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.find("Bias") != std::string::npos) {
            // Start a new layer
            if (!current_layer.empty()) {
                biases.push_back(current_layer);
                current_layer.clear();
            }
        } else {
            // Parse biases for the current layer
            std::istringstream iss(line);
            double value;
            while (iss >> value) {
                current_layer.push_back(value);
            }
        }
    }

    // Add the last layer
    if (!current_layer.empty()) {
        biases.push_back(current_layer);
    }

    file.close();
    return biases;
}


///*
///*
Affine2Vector neuron(const double B, const Affine2Vector& In_aff) {
    //sigmoid----------------------------------
    IntervalVector yinit(2);
    yinit[0] = Interval(0.5, 0.5);
    yinit[1] = Interval(0, 0);
    Affine2Vector yinit_aff(yinit, true);

    IntervalVector Output(1);
    Output[0] = Interval(0, 0);
        Affine2Vector Output_aff(Output, true);
  yinit_aff[1] = In_aff[0];
  Variable y(2);

    // Set affine noise number
    AF_fAFFullI::setAffineNoiseNumber(50); //regle la precision en évitant l'explosion des boites
    //450 ça marche bien

        //(1 neuron sigmoid-like behavior)
        Function ydot = Function(y, Return(
            (y[1] + B) * y[0] * (1 - y[0]),
            0 * y[1]
        ));

        // Create the ODE problem
        ivp_ode problem = ivp_ode(ydot, 0, yinit_aff, SYMBOLIC);
        simulation simu = simulation(&problem, 1, HEUN, 1e-5, 1e-2);
        simu.run_simulation();
        simu.get_tight(0.5);
        yinit_aff = simu.get_last_aff();
        Output_aff[0] = yinit_aff[0];
    // Return the final reachable set
        nb_sigmoid++;
    //std::cout << "Sigmoid--"<< nb_sigmoid << Output_aff << std::endl;
    return Output_aff;
}
//*/
/*
Affine2Vector neuron(const double B, const Affine2Vector& In_aff) {
    //exp----------------------------------
    IntervalVector yinit(2);
    yinit[0] = Interval(0.5, 0.5);
    yinit[1] = Interval(0, 0);
    Affine2Vector yinit_aff(yinit, true);

    IntervalVector Output(1);
    Output[0] = Interval(0, 0);
        Affine2Vector Output_aff(Output, true);

  yinit_aff[1] = In_aff[0];
  Variable y(2);

    // Set affine noise number
    AF_fAFFullI::setAffineNoiseNumber(25); //regle la precision en évitant l'explosion des boites


        //(1 neuron sigmoid-like behavior)
        Function ydot = Function(y, Return(
            (y[1]+B)*y[0],
            0 * y[1]
        ));

        // Create the ODE problem
        ivp_ode problem = ivp_ode(ydot, 0, yinit_aff, SYMBOLIC);
        simulation simu = simulation(&problem, 1, HEUN, 1e-7, 1e-2);
        simu.run_simulation();

        yinit_aff = simu.get_last_aff();
        Output_aff[0] = yinit_aff[0];
    // Return the final reachable set
        nb_sigmoid++;
    std::cout << "Sigmoid--"<< nb_sigmoid << Output_aff << std::endl;
    return Output_aff;
}
*/

//*/
/*
Affine2Vector neuron(const double B, const Affine2Vector& In_aff) {
    AF_fAFFullI::setAffineNoiseNumber(250); //regle la precision en évitant l'explosion des boites
    //tan hyperbolic--------------------------------
    IntervalVector yinit(2);
    yinit[0] = Interval(0, 0);
    yinit[1] = Interval(0, 0);
    Affine2Vector yinit_aff(yinit, true);

    IntervalVector Output(1);
    Output[0] = Interval(0, 0);
        Affine2Vector Output_aff(Output, true);

  yinit_aff[1] = In_aff[0];

    Variable y(2);
        // Define the dynamics of the system (1 neuron tan hyperbolic-like behavior)
        Function ydot = Function(y, Return(
            (y[1] + B) * (1 - y[0]*y[0]),
            0 * y[1]
        ));

        // Create the ODE problem with symbolic handling
        ivp_ode problem = ivp_ode(ydot, 0, yinit_aff, SYMBOLIC);
        simulation simu = simulation(&problem, 1, HEUN, 1e-7,1e-2);
        simu.run_simulation();

        yinit_aff = simu.get_last_aff();
        Output_aff[0] = yinit_aff[0];
    // Return the final reachable set
    std::cout << "TanH--" << nb_sigmoid << Output_aff <<std::endl;
    nb_sigmoid++;
    return Output_aff;
}
*/

Affine2Vector neuron_softplus(const double B, const Affine2Vector& In_aff) {
    //softplus--------------------------------
    IntervalVector yinit(3);
    yinit[0] = Interval(0.5, 0.5);
    yinit[1] = Interval(0, 0);
    yinit[2] = Interval(0.3010299957,0.3010300000); //log(2) approximation
    Affine2Vector yinit_aff(yinit, true);

    IntervalVector Output(1);
    Output[0] = Interval(0, 0);
        Affine2Vector Output_aff(Output, true);

  yinit_aff[1] = In_aff[0];
  Variable y(3);

    // Set affine noise number
    AF_fAFFullI::setAffineNoiseNumber(60);


        // Define the dynamics of the system (1 neuron softmax-like behavior)
        Function ydot = Function(y, Return(
            (y[1] + B) * y[0] * (1 - y[0]),
            0 * y[1],
            (y[1] + B)*y[0]
        ));

        // Create the ODE problem with symbolic handling
        ivp_ode problem = ivp_ode(ydot, 0, yinit_aff, SYMBOLIC);
        simulation simu = simulation(&problem, 1, HEUN, 1e-7,0.1);
        simu.run_simulation();

        yinit_aff = simu.get_last_aff();
        Output_aff[0] = yinit_aff[2];
    // Return the final reachable set
    std::cout << "softplus--" << std::endl;
    //nb_sigmoid++;
    return Output_aff;
}

std::vector<Affine2Vector> aggregate_inputs(const std::vector<Affine2Vector>& inputs, const std::vector<std::vector<double>>& weights) {
    size_t n_neurons = weights.size(); // Number of neurons in the layer
    //std::cout << "number of neurons: " << n_neurons << std::endl;
    size_t n_input = weights[0].size(); //à virer
    //std::cout << "number of input: " << n_input << ", input last layer: " << inputs.size() << std::endl;
    std::vector<Affine2Vector> aggregated_inputs(n_neurons, 0 * inputs[0]); // Initialize to zero Affine2Vector
/*
for (size_t j = 0; j < inputs.size(); ++j) {
    std::cout << "Input: " << j << ": " << inputs[j] << std::endl;
}
*/
    for (size_t neuron = 0; neuron < n_neurons; ++neuron) {
        for (size_t input_idx = 0; input_idx < n_input; ++input_idx) {
            if (weights[neuron][input_idx]!=0) //si on fait du pruning on skip la somme d'une multiplication par un poid nulle
            {
            aggregated_inputs[neuron] = aggregated_inputs[neuron] + weights[neuron][input_idx] * inputs[input_idx];
            }
        }
         //std::cout << neuron << std::endl;
         aggregated_inputs[neuron].compact(1e-15);
    }

    //std::cout << "aggregate_inputs output size: " << aggregated_inputs.size() << std::endl;
for (size_t i = 0; i < aggregated_inputs.size(); ++i) {
    //std::cout << "Neuron Sum Wij*Xi, i = " << i << ": " << aggregated_inputs[i] << std::endl;
}
//std::cout << "-------------------------------" << std::endl;
    return aggregated_inputs;
}


std::vector<Affine2Vector> softplus_neural_layer(const std::vector<Affine2Vector>& inputs, const std::vector<std::vector<double>>& weights, const std::vector<double>& offsets) {
    std::vector<Affine2Vector> outputs;

    // Compute the aggregated inputs for all neurons
    std::vector<Affine2Vector> aggregated_inputs = aggregate_inputs(inputs, weights);

    // Compute outputs for each neuron
    for (size_t i = 0; i < aggregated_inputs.size(); ++i) {
        //std::cout << "--Neuron: "<< i << std::endl;
        Affine2Vector output = neuron_softplus(offsets[i], aggregated_inputs[i]);
        outputs.push_back(output);
    }

    return outputs;
}


std::vector<Affine2Vector> neural_layer(const std::vector<Affine2Vector>& inputs, const std::vector<std::vector<double>>& weights, const std::vector<double>& offsets) {
    std::vector<Affine2Vector> outputs;

    // Compute the aggregated inputs for all neurons
    std::vector<Affine2Vector> aggregated_inputs = aggregate_inputs(inputs, weights);

    // Compute outputs for each neuron
    for (size_t i = 0; i < aggregated_inputs.size(); ++i) {
        //std::cout << "--Neuron: "<< i << std::endl;
        Affine2Vector output = neuron(offsets[i], aggregated_inputs[i]);
        outputs.push_back(output);
    }

    return outputs;
}

/*
std::vector<Affine2Vector> linear_output_layer(const std::vector<Affine2Vector>& inputs, const std::vector<std::vector<double>>& weights, const std::vector<double>& offsets) {
    std::vector<Affine2Vector> outputs;

    // Compute the aggregated inputs for all output neurons
    std::vector<Affine2Vector> aggregated_inputs = aggregate_inputs(inputs, weights);

    // Add offsets to the aggregated inputs for each output neuron
    for (size_t i = 0; i < aggregated_inputs.size(); ++i) {
        //std::cout << "Sum: WjiXi " << i <<" :"<< aggregated_inputs[i] << std::endl;
        Affine2Vector output = aggregated_inputs[i] + offsets[i];
        //std::cout << "Sum: WjiXi + B :"<< output << std::endl;
        outputs.push_back(output);
    }

    return outputs;
}
*/
///*
std::vector<Affine2Vector> linear_output_layer(const std::vector<Affine2Vector>& inputs, const std::vector<std::vector<double>>& weights, const std::vector<double>& offsets) {
    std::vector<Affine2Vector> outputs;

    // Initialize Affine2Vector for the offsets
    int n = offsets.size();  // Number of neurons in the output layer
    IntervalVector offset_intervals(n);
    for (int i = 0; i < n; ++i) {
        offset_intervals[i] = Interval(offsets[i], offsets[i]);  // Set each offset as a fixed interval
    }
    Affine2Vector offset_affine(offset_intervals, true);  // Create an Affine2Vector from the intervals


        IntervalVector temp_offset(1); ///temporary interval vector
        temp_offset[0] = offset_intervals[0];
        Affine2Vector temp_offset_affine(temp_offset, true);

    // Compute the aggregated input using the helper function
    std::vector<Affine2Vector> aggregated_inputs = aggregate_inputs(inputs, weights);
    // Compute outputs for each neuron in the output layer (linear function)
    for (size_t i = 0; i < n; ++i) {
        temp_offset[0] = offset_intervals[i];
        Affine2Vector temp_offset_affine(temp_offset, true);

        // Add the aggregated input and the offset
        Affine2Vector output = aggregated_inputs[i] + temp_offset_affine;
        outputs.push_back(output);
    }

    return outputs;
}
//*/
std::pair<std::vector<std::vector<double>>, std::vector<double>> affine2singleton(
    const std::vector<std::vector<std::vector<double>>>& weights,
    const std::vector<std::vector<double>>& biases,
    const std::vector<Affine2Vector>& inputs) 
{
    std::vector<std::vector<double>> outputWeights(weights[0].size(), std::vector<double>());  // Initialize with the correct size
    std::vector<double> outputbiases = biases[0];
    std::vector<double> sum_diff(biases[0].size(), 0);
    std::vector<int> temp_list;
    std::vector<int> anti_temp_list;

    // Identify singleton and interval inputs
    for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i][0].itv().lb() == inputs[i][0].itv().ub()) {
            temp_list.push_back(i);
            //std::cout << "Singleton index: " << i << std::endl;
        } else {
            anti_temp_list.push_back(i);
            //std::cout << "Interval index: " << i << std::endl;
        }
    }

    // Compute bias updates for singleton inputs
    if (!temp_list.empty()) {
        for (size_t j = 0; j < sum_diff.size(); j++) {
            for (size_t i = 0; i < temp_list.size(); i++) {
                size_t singleton_idx = temp_list[i];
                double singleton_value = inputs[singleton_idx][0].itv().ub();
                sum_diff[j] += weights[0][j][singleton_idx] * singleton_value;
            }
        }

        for (size_t i = 0; i < sum_diff.size(); i++) {
            outputbiases[i] += sum_diff[i];
            //std::cout << "Updated bias " << i << ": " << outputbiases[i] << std::endl;
        }
    }

    // Construct outputWeights for non-singleton inputs
    if (!anti_temp_list.empty()) {
        for (size_t i = 0; i < weights[0].size(); i++) {
            for (size_t j = 0; j < anti_temp_list.size(); j++) {
                size_t interval_idx = anti_temp_list[j];
                outputWeights[i].push_back(weights[0][i][interval_idx]);
            }
        }
    } else {
        outputWeights = weights[0];  // No singleton inputs, retain original weights
    }

    // Return updated weights and biases
    return {outputWeights, outputbiases};
}


std::vector<Affine2Vector> neural_network_hidden(const std::vector<std::vector<std::vector<double>>>& weights, const std::vector<std::vector<double>>& offsets, const std::vector<Affine2Vector>& input) {
    std::vector<Affine2Vector> current_inputs = input;

    for (size_t layer = 0; layer < weights.size(); ++layer) {
        if (layer == weights.size() - 1) {
            // Use the linear output layer for the last layer
            current_inputs = linear_output_layer(current_inputs, weights[layer], offsets[layer]);
            //current_inputs = softplus_neural_layer(current_inputs, weights[layer], offsets[layer]);
        } else {
            // Use the normal hidden layer for other layers
            current_inputs = neural_layer(current_inputs, weights[layer], offsets[layer]);
        }
    }

    return current_inputs;
}

std::vector<Affine2Vector> inputs2singleton (const std::vector<Affine2Vector>& inputs)
{
   std::vector<int> anti_temp_list;
    std::vector<Affine2Vector> output;
    // Identify singleton and interval inputs
    for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i][0].itv().lb() != inputs[i][0].itv().ub()) {
            anti_temp_list.push_back(i);
            //std::cout << "Interval index: " << i << std::endl;
        }
    }
    // output stay in the same order as initial one
    // Compute bias updates for singleton inputs
    if (!anti_temp_list.empty()) {
        for (size_t j = 0; j < anti_temp_list.size(); j++) {
            output.push_back(inputs[anti_temp_list[j]]);
        }
    }
    else
    {
        output = inputs;
    }
    return output;
}
/*
std::vector<std::vector<Affine2Vector>> Affine2Splitter(const std::vector<Affine2Vector>& inputs,const int& n_pavage) //pavage de l'entrée
{

    std::vector<Affine2Vector> formated_inputs = inputs2singleton(inputs);
    std::vector<std::pair<double,int>> radius;
    
    std::vector<std::vector<Affine2Vector>> output(n_pavage*n_pavage,inputs);
    //std::vector<std::vector<Affine2Vector>> output;
    IntervalVector temp_xy(2);

    if (formated_inputs.size()>2)//if >2d renvoyer direct l'entrée dans un {inputs} bissection uniquement en 2d
    {
        return {inputs};
    }
    
          for (size_t i = 0; i < formated_inputs.size(); i++)
      {
        std::cout<<"--formated "<<i<<": "<<formated_inputs[i]<<std::endl;
      }

    for (size_t i = 0; i < 2; i++)// en 2 dimensions
    {
        radius.push_back({formated_inputs[i][0].itv().ub()-formated_inputs[i][0].itv().lb(),i});

        if (formated_inputs[i][0].itv().lb()=>0)
        {
            temp_xy[i] = Interval(0, formated_inputs[i][0].itv().lb());
        }
        else
        {
            temp_xy[i] = Interval(0, formated_inputs[i][0].itv().lb());
        }
        temp_xy[i] = Interval(0, formated_inputs[i][0].itv().lb());
        std::cout<<formated_inputs[i][0].itv().lb()<<std::endl;
    }

   // temp_xy[0] = Interval(0, formated_inputs[0][0].itv().lb());
   // temp_xy[1] = Interval(0, formated_inputs[1][0].itv().lb());
    std::cout<< temp_xy[0] << " |temp1| " << temp_xy[1] <<std::endl;
    for (size_t j = 0; j < n_pavage; j++)
    {   
        temp_xy[0] = Interval(0, formated_inputs[0][0].itv().lb());
        temp_xy[1] = Interval(temp_xy[1].ub(), temp_xy[1].ub()+radius[1].first*1/n_pavage);   
        for (size_t i = 0; i < n_pavage; i++)
        {
            temp_xy[0] = Interval(temp_xy[0].ub(), temp_xy[0].ub()+radius[0].first*1/n_pavage);
            //temp_xy[1] = Interval(temp_xy[1].ub(), temp_xy[1].ub()+radius[1].first*1/n_pavage);           
            std::cout<< radius[0].first<< "|radius|" << radius[1].first<<std::endl;
            std::cout<< temp_xy[0] << " |temp_loop| " << temp_xy[1] <<std::endl;
            Affine2Vector temp_xy_aff(temp_xy, true);
            output[i+n_pavage*j][0][0] = temp_xy_aff[0];
            output[i+n_pavage*j][1][0] = temp_xy_aff[1];

            //std::cout<< temp_xy_aff[0] << " |temp| " << temp_xy_aff[1] <<std::endl;
            std::cout<< output[i+n_pavage*j][0][0]<<" ; "<< output[i+n_pavage*j][1][0] <<" | "<< i+n_pavage*j <<std::endl;
        }
    }

    return output;


    //calculer la largeur sur chaque axe
    //paver la boite en N*N boite
    // sortir un vecteur avec les N*Ns boites
    // faire une boucle for dans l'implem me permettant de parcourir tout les vecteurs. ensuite il faut faire l'union de toutes les solutions.

}
*/
std::vector<std::vector<Affine2Vector>> Affine2Splitter(const std::vector<Affine2Vector>& inputs, const int& n_pavage) {
    std::vector<Affine2Vector> formated_inputs = inputs2singleton(inputs);
    std::vector<std::pair<double, int>> radius;
    std::vector<std::vector<Affine2Vector>> output(n_pavage * n_pavage, inputs);
    IntervalVector temp_xy(2);

    if (formated_inputs.size() > 2) {
        return {inputs};  // Return directly if more than 2D
    }

    // Print formatted inputs
    /*
    for (size_t i = 0; i < formated_inputs.size(); i++) {
        std::cout << "--formated " << i << ": " << formated_inputs[i] << std::endl;
    }
    */
    // Compute radius for each dimension
    for (size_t i = 0; i < 2; i++) {
        double lb = formated_inputs[i][0].itv().lb();
        double ub = formated_inputs[i][0].itv().ub();
        radius.push_back({ub - lb, i});
    }

    // Fix initial interval bounds
    temp_xy[0] = Interval(formated_inputs[0][0].itv().lb(), formated_inputs[0][0].itv().ub());
    temp_xy[1] = Interval(formated_inputs[1][0].itv().lb(), formated_inputs[1][0].itv().ub());
    
    //std::cout << temp_xy[0] << " |temp1| " << temp_xy[1] << std::endl;

    // Iterate over grid
    for (size_t j = 0; j < n_pavage; j++) {
        double y_start = formated_inputs[1][0].itv().lb();
        double y_step = radius[1].first / n_pavage;
        double y_lb = y_start + j * y_step;
        double y_ub = y_start + (j + 1) * y_step;

        for (size_t i = 0; i < n_pavage; i++) {
            double x_start = formated_inputs[0][0].itv().lb();
            double x_step = radius[0].first / n_pavage;
            double x_lb = x_start + i * x_step;
            double x_ub = x_start + (i + 1) * x_step;

            // Update interval bounds correctly
            temp_xy[0] = Interval(x_lb, x_ub);
            temp_xy[1] = Interval(y_lb, y_ub);

            //std::cout << radius[0].first << "|radius|" << radius[1].first << std::endl;
            //std::cout << temp_xy[0] << " |temp_loop| " << temp_xy[1] << std::endl;

            Affine2Vector temp_xy_aff(temp_xy, true);

            // Properly store into output (ensuring the index is valid)
            output[i+n_pavage*j][0][0] = temp_xy_aff[0];
            output[i+n_pavage*j][1][0] = temp_xy_aff[1];

            //std::cout << output[i + n_pavage * j][0][0] << " ; " << output[i + n_pavage * j][1][0] << " | " << i + n_pavage * j << std::endl;
        }
    }

    return output;

        //calculer la largeur sur chaque axe
    //paver la boite en N*N boite
    // sortir un vecteur avec les N*Ns boites
    // faire une boucle for dans l'implem me permettant de parcourir tout les vecteurs. ensuite il faut faire l'union de toutes les solutions.

}


std::vector<std::vector<Affine2Vector>> Affine2Bisec(const std::vector<Affine2Vector>& inputs) {
    std::vector<Affine2Vector> formated_inputs = inputs2singleton(inputs);
    std::vector<std::vector<Affine2Vector>> output(4, inputs);
    IntervalVector temp_xy(2);

    if (formated_inputs.size() > 2) {
        return {inputs};  // Return directly if more than 2D
    }
    output = Affine2Splitter(formated_inputs, 2);
//on fait pas vraiment une disection mais on fait un pavage adaptatif
    return output;
}

 std::vector<Affine2Vector> DeepNeuralNetwork_aff(const std::vector<Affine2Vector>& inputs, const std::string& weightsFile, const std::string& biasesFile){
    // File paths
    std::vector<std::vector<std::vector<double>>> weights = readWeights(weightsFile);
    std::vector<std::vector<double>> biases = readBiases(biasesFile);
    std::vector<std::vector<Affine2Vector>> paved_inputs;
    //std::vector<std::vector<Affine2Vector>> final_reachable_states;

        // Call affine2singleton
    auto first_layer = affine2singleton(weights, biases, inputs); //when an input is a singleton it will treat it as a scalar to avoid unnecessary null noise number

    // Update weights and biases for the network
    weights[0] = first_layer.first;
    biases[0] = first_layer.second;
    //printWeightsAndBiases(weights,biases);

    auto transformed_input = inputs2singleton(inputs); //when an input is a singleton it will treat it as a scalar to avoid unnecessary null noise number   
    std::vector<Affine2Vector> final_reachable_states = neural_network_hidden(weights, biases, transformed_input);
  nb_sigmoid = 0;
  return final_reachable_states;
 }
