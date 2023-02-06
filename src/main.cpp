#include "main.hpp"

int main()
{
    unsigned int n, n_test;
    int m = 10;
    image *training_images = read_images(TRAINING_IMAGES, &n);
    byte *training_labels = read_labels(TRAINING_LABELS, &n);
    image *test_images = read_images(TEST_IMAGES, &n_test);
    byte *test_labels = read_labels(TEST_LABELS, &n_test);
    MCIA::Layer layer1(784, 30, &MCIA::activation_function, &MCIA::d_activation_function, 0.05, m);
    MCIA::Layer layer2(30, 10, &MCIA::activation_function, &MCIA::d_activation_function, 0.05, m);
    MCIA::Network network(std::vector<MCIA::Layer>{layer1, layer2}, m);

    MCIA::set *training_set = new MCIA::set(n);
    MCIA::set *test_set = new MCIA::set(n_test);

    for (int indx = 0; indx < n; ++indx)
    {
        Eigen::Map<Eigen::Matrix<byte, 784, 1>> temp(&(training_images[indx][0]));
        (*training_set)[indx] = std::pair<Eigen::MatrixXd, byte>(temp.cast<double>() / 255.0, training_labels[indx]);
    }
    for (int indx = 0; indx < n_test; ++indx)
    {
        Eigen::Map<Eigen::Matrix<byte, 784, 1>> temp(&(test_images[indx][0]));
        (*test_set)[indx] = std::pair<Eigen::MatrixXd, byte>(temp.cast<double>() / 255.0, test_labels[indx]);
    }

    free(training_images);
    free(training_labels);
    free(test_images);
    free(test_labels);

    {
        std::cout << "REFERENCE" << std::endl;
        float correct = 0;
        for (auto &pair : (*test_set))
        {
            if (network.result(pair.first) == pair.second)
            {
                correct += 1;
            }
        }
        std::cout << "correctness : " << correct / float(n_test) * 100.0 << " %" << std::endl;
    }

    for (int epoch = 1; epoch <= 30; ++epoch)
    {
        std::random_shuffle(training_set->begin(), training_set->end());
        for (int batch = 0; batch * m < n; ++batch)
        {
            Eigen::MatrixXd images = Eigen::MatrixXd(784, m);
            Eigen::VectorXd labels = Eigen::VectorXd(m);
            for (int indx = 0; indx < m; ++indx)
            {
                images.col(indx) = (*training_set)[m * batch + indx].first.col(0);
                labels(indx) = static_cast<double>((*training_set)[m * batch + indx].second);
            }
            network.learn(images, labels);
        }
        std::cout << "AFTER EPOCH " << epoch << std::endl;
        float correct = 0;
        for (auto &pair : (*test_set))
        {
            if (network.result(pair.first) == pair.second)
            {
                correct += 1;
            }
        }
        std::cout << "correctness : " << correct / float(n_test) * 100.0 << " %" << std::endl;
    }

    delete training_set;
    delete test_set;

    return 0;
}