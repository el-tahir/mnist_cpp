#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <ctime>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <chrono>

#include "Matrix.h"
#include "NeuralNetwork.h"
#include "utils.h"


void start_server(NeuralNetwork& nn, int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == 0) {
        throw std::runtime_error("socket creation failed");
    }

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        throw std::runtime_error("bind failed");
    }

    if (listen(server_fd, 3) < 0) {
        throw std::runtime_error("listen failed");
    }

    std::cout << "server listening on port " << port << "..." << std::endl;

    while(true) {
        int new_socket = accept(server_fd, nullptr, nullptr);
        if (new_socket < 0) {
            std::cerr << "accept failed" << std::endl;
            continue;
        }

        std::vector<float> buffer(784);

        int bytes_read = read(new_socket, buffer.data(), 784 * sizeof(float));

        if (bytes_read == 784 * sizeof(float)) {
            Matrix input(1, 784);
            input.data = buffer;

            //make prediction
            auto cache = nn.forward(input.T());
            auto predictions = get_predictions(cache.A2);
            int result = predictions[0];

            std::cout << "received request. prediction : " << result << std::endl;

            print_image(input, 0);
            
            // send back the prediction

            std::string response = std::to_string(result);

            write(new_socket, response.c_str(), response.size());
        
        }

        close(new_socket);

    }
}



int main() {

    std::srand(std::time(0));

    Matrix test_images = read_mnist_images("t10k-images-idx3-ubyte");
    std::vector<int> test_labels = read_mnist_labels("t10k-labels-idx1-ubyte");

    std::string filename = "model.bin";
    NeuralNetwork nn(784, 128, 10);

    try {
        nn.load(filename);
        std::cout << "model loaded successfully" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "no saved model found, training from scratch..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        Matrix train_images = read_mnist_images("train-images-idx3-ubyte");
        std::vector<int> train_labels = read_mnist_labels("train-labels-idx1-ubyte");
    
        int epochs = 20;
        int batch_size = 64;
        int num_samples = train_images.rows;
        int num_batches = (num_samples + batch_size - 1) / batch_size;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batch = 0; batch < num_batches; batch++) {
                int start = batch * batch_size;
                int end = std::min(start + batch_size, num_samples);
                int current_batch_size = end - start;
                
                Matrix X_batch (current_batch_size, 784);
                for (int i = 0; i < current_batch_size; i++) {
                    for (int j = 0; j < 784; j++) {
                        X_batch(i, j) = train_images(start + i, j);
                    }
                }
                
                std::vector<int> Y_batch;
                Y_batch.reserve(current_batch_size);
                for (int i = start; i < end; i++) {
                    Y_batch.push_back(train_labels[i]);
                }
                
                Matrix X_batch_T = X_batch.T();
                auto cache = nn.forward(X_batch_T);
                nn.backward(X_batch_T, Y_batch, cache);
            }
            
            auto cache = nn.forward(test_images.T());
            auto predictions = get_predictions(cache.A2);
            float acc = get_accuracy(predictions, test_labels);
            std::cout << "Epoch " << epoch << " accuracy: " << acc << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        std::cout << "training completed in " << duration << " seconds." << std::endl;
        
        nn.save(filename);
    }

    std::cout << "\n--- verification ---\n";
    for (int i = 0; i < 5; i++) {
        int random = rand() % test_images.rows;
        print_image(test_images, random);
        Matrix input = Matrix(1, 784);
        for (int j = 0; j < 784; j++) {
            input(0, j) = test_images(random, j);
        }
        auto cache = nn.forward(input.T());
        auto prediction = get_predictions(cache.A2);
        
        std::cout << "Predicted: " << prediction[0] 
                << " Actual: " << test_labels[random] << std::endl;
    }

    std::cout << "starting server...." << std::endl;
    start_server(nn, 8080);
    return 0;
}


