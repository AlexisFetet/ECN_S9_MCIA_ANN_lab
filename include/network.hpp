#ifndef NETWORK_H
#define NETWORK_H

#include <Eigen/Dense>
#include <layer.hpp>
#include <vector>

namespace MCIA
{
    class Network
    {
    public:
        Network(std::vector<Layer> layers, int m);
        ~Network();
        Eigen::MatrixXd compute(Eigen::MatrixXd input_vector);
        void learn(Eigen::MatrixXd image, Eigen::VectorXd expected);
        int result(Eigen::MatrixXd image);

    private:
        Eigen::MatrixXd m_eye;
        std::vector<Layer> m_layers;
        int m_m;
    };
}

#endif