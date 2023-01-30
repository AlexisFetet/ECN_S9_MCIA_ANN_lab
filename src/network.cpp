#include <network.hpp>

namespace MCIA
{
    Network::Network(std::vector<Layer> layers, int m)
    {
        m_m = m;
        m_eye = Eigen::MatrixXd::Identity(layers.back().get_weight_cols(), layers.back().get_weight_cols());
        m_layers = layers;
    }

    Network::~Network() {}

    Eigen::MatrixXd Network::compute(Eigen::MatrixXd input_vector)
    {
        for (auto &layer : m_layers)
        {
            input_vector = layer.forward(input_vector);
        }
        return input_vector;
    }

    void Network::learn(Eigen::MatrixXd image, Eigen::VectorXd expected)
    {
        Eigen::MatrixXd result = compute(image);
        Eigen::MatrixXd expected_result = Eigen::MatrixXd::Zero(result.rows(), result.cols());
        for (std::size_t indx = 0; indx < expected.size(); ++indx)
        {
            expected_result((int)expected[indx], (int)indx) = 1;
        }
        Eigen::MatrixXd delta = result - expected_result;
        for (std::size_t indx = m_layers.size() - 1; indx >= 0; --indx)
        {
            if (indx == m_layers.size() - 1)
            {
                delta = m_layers[indx].backward(delta, m_eye);
            }
            else
            {
                delta = m_layers[indx].backward(delta, m_layers[indx + 1].get_weight());
            }
        }
        for (auto &layer : m_layers)
        {
            layer.update();
        }
    }

    int Network::result(Eigen::MatrixXd image)
    {
        Eigen::MatrixXd output = compute(image);
        int i, j;
        output.maxCoeff(&i, &j);
        return i;
    }
}