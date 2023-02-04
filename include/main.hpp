#ifndef MAIN_HPP
#define MAIN_HPP

#include <network.hpp>
#include <parser.h>
#include <utility>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace MCIA
{
    typedef std::vector<std::pair<Eigen::MatrixXd, byte>> set;
    double activation_function(double input) { return 1 / (1 + exp(-1*input)); };
    double d_activation_function(double input)
    {
        double temp = activation_function(input);
        return temp * (1 - temp);
    };
}

#endif