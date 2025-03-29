#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct XOR
{
    float And_w1;
    float And_w2;
    float And_b;

    float Or_w1;
    float Or_w2;
    float Or_b;

    float Nand_w1;
    float Nand_w2;
    float Nand_b;
} XOR ;

float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

typedef float simple[3];
int train_count = 4;

simple Xor_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0}
};

float forward(XOR m, float x1, float x2) {
    float a1 = sigmoid(m.Or_w1*x1 + m.Or_w2*x2 + m.Or_b);
    float a2 = sigmoid(m.Nand_w1*x1 + m.Nand_w2*x2 + m.Nand_b);
    return sigmoid(m.And_w1*a1 + m.And_w2*a2 + m.And_b);
}

simple* train = Xor_train;

float cost(XOR m) {
    float result = 0.0f;
    for (int i = 0; i < train_count; ++i) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m, x1, x2);
        float d = y - train[i][2];
        result += d * d;
    }
    result /= train_count;

    return result;
}

float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

XOR Init_rand() {
    XOR m;
    m.Or_w1   = rand_float();
    m.Or_w2   = rand_float();
    m.Or_b    = rand_float();
    m.Nand_w1 = rand_float();
    m.Nand_w2 = rand_float();
    m.Nand_b  = rand_float();
    m.And_w1  = rand_float();
    m.And_w2  = rand_float();
    m.And_b   = rand_float();

    return m;
}

XOR learn(XOR m, XOR g, float rate) {
    m.Or_w1   -= rate*g.Or_w1;
    m.Or_w2   -= rate*g.Or_w2;
    m.Or_b    -= rate*g.Or_b;
    m.Nand_w1 -= rate*g.Nand_w1;
    m.Nand_w2 -= rate*g.Nand_w2;
    m.Nand_b  -= rate*g.Nand_b;
    m.And_w1  -= rate*g.And_w1;
    m.And_w2  -= rate*g.And_w2;
    m.And_b   -= rate*g.And_b;

    return m;
}

XOR derive(XOR m, float esp) {
    XOR g;
    float c = cost(m);
    float saved;

    saved = m.Or_w1;
    m.Or_w1 += esp;
    g.Or_w1 = (cost(m) - c) / esp;
    m.Or_w1 = saved;

    saved = m.Or_w2;
    m.Or_w2 += esp;
    g.Or_w2 = (cost(m) - c) / esp;
    m.Or_w2 = saved;
    
    saved = m.Or_b;
    m.Or_b += esp;
    g.Or_b = (cost(m) - c) / esp;
    m.Or_b = saved;

    saved = m.Nand_w1;
    m.Nand_w1 += esp;
    g.Nand_w1 = (cost(m) - c) / esp;
    m.Nand_w1 = saved;

    saved = m.Nand_w2;
    m.Nand_w2 += esp;
    g.Nand_w2 = (cost(m) - c) / esp;
    m.Nand_w2 = saved;

    saved = m.Nand_b;
    m.Nand_b += esp;    
    g.Nand_b = (cost(m) - c) / esp;
    m.Nand_b = saved;

    saved = m.And_w1;
    m.And_w1 += esp;
    g.And_w1 = (cost(m) - c) / esp;
    m.And_w1 = saved;

    saved = m.And_w2;
    m.And_w2 += esp;
    g.And_w2 = (cost(m) - c) / esp;
    m.And_w2 = saved;

    saved = m.And_b;
    m.And_b += esp;
    g.And_b = (cost(m) - c) / esp;
    m.And_b = saved;

    return g;
}

int main() {
    srand(time(0));
    XOR m = Init_rand();

    float esp = 1e-1;
    float rate = 1e-1;

    for (int i = 0; i < 1000 * 100; ++i) {
        XOR g = derive(m, esp);
        m = learn(m, g, rate);
    }
    printf("cost = %f\n", cost(m));

    printf("\n---------------------XOR--------------------\n");
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            printf("%zu ^ %zu = %f\n", i, j, forward(m, i, j));
        }
    }

    return 0;
}