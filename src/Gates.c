#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

typedef float simple[3];

// AND-Gate
simple And_train[] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1}
};

// OR-Gate
simple Or_train[] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1}
};

// NAND-Gate
simple Nand_train[] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0}
};

// NOR-Gate
simple Nor_train[] = {
    {0, 0, 1},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 0}
};

simple* train = And_train;
int train_count = 4;

float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

float cost(float w1, float w2, float b) {
    float result = 0.0f;
    for (int i = 0; i < train_count; ++i) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoid(x1 * w1 + x2 * w2 + b);
        float d = y - train[i][2];
        result += d * d;
    }
    result /= train_count;

    return result;
}

int main() {
    srand(time(NULL));
    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();

    float eps = 1e-3;
    float rate = 1e-3;

    for (int i = 0; i < 1000 * 1000; ++i) {
        float dw1 = (cost(w1 + eps, w2, b) - cost(w1, w2, b))/eps;
        float dw2 = (cost(w1, w2 + eps, b) - cost(w1, w2, b))/eps;
        float db = (cost(w1, w2, b + eps) - cost(w1, w2, b))/eps;
        w1 -= dw1 * rate;
        w2 -= dw2 * rate;
        b -= db * rate;
    }
    printf("W1: %f | W2: %f | C: %f\n", w1, w2, cost(w1, w2, b));

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            printf("%zu | %zu = %f\n", i, j, sigmoid(i*w1 + j*w2 + b));
        }
    }
    
    return 0;
}