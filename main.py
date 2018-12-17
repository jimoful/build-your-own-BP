#coding:utf-8
import random
import math
import matplotlib.pyplot as plt
import  numpy as np

# 参数解释：
# "pd_" ：偏导的前缀
# "d_" ：导数的前缀
# "w_ho" ：隐含层到输出层的权重系数索引
# "w_ih" ：输入层到隐含层的权重系数的索引


###############################################
# class Neuron for one neuron                 #
###############################################


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.inputs = []
        self.output = 0

    # 单个神经元所有输入与权重乘积之和 加上bias
    def calculate_total_neuron_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # 激活函数
    @staticmethod
    def neu_sigmod(total_neuron_input):
        return 1/(1+math.exp(-total_neuron_input))

    # 单个神经元的输出
    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.neu_sigmod(self.calculate_total_neuron_input())
        return self.output

    # 计算单个神经元误差
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # 对输出的偏导
    def calculate_pd_error_out(self, target_output):
        return (target_output - self.output) * (-1)

    # 对sigmod的偏导
    def calculate_pd_error_sigmod(self):
        return self.output * (1.0 - self.output)

###############################################
# class NeuronLayer for one layer of neuron   #
###############################################


class NeuronLayer:
    def __init__(self, num_neurons, weights, num_weights_per_neurons, bias):
        # 同一层神经元共享一个bias
        self.bias = bias if bias else random.random()
        self.neurons = []
        self.outputs = []

        for i in range(num_neurons):
            self.neurons.append(Neuron(bias))

        # 初始权值
        self.init_input_weights(weights, num_weights_per_neurons)

    # 打印一层神经元的权重
    def inspect(self):
        for i in range(len(self.neurons)):
            print('Neuron:', i)
            for j in range(len(self.neurons[i].weights)):
                print(' Weight:', self.neurons[i].weights[j])
            print('Bias:', self.bias)

    # 前向传递，计算一层
    def feed_forward(self, inputs):
        self.outputs = []
        for neuron in self.neurons:
            self.outputs.append(neuron.calculate_output(inputs))
        return self.outputs

    # 初始化一层神经元输入权值
    def init_input_weights(self, weights, num_weights_per_neurons):
        index = 0
        for i in range(len(self.neurons)):
            for j in range(num_weights_per_neurons):
                if not weights:
                    self.neurons[i].weights.append(random.random())
                else:
                    self.neurons[i].weights.append(weights[index])
                    index += 1



###############################################
# class NeuralNetwork for the whole network   #
###############################################


class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights=None, hidden_layer_bias=None, output_layer_weights=None,
                 output_layer_bias=None):

        self.num_inputs = num_inputs
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_weights, num_inputs, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_weights, num_hidden, output_layer_bias)
        # self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        # self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    # 初始化隐藏层的输入权重
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        index_weight = 0
        for i in range(len(self.hidden_layer.neurons)):
            for j in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[i].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[i].weights.append(hidden_layer_weights[index_weight])
                index_weight += 1

    # 初始化输出层权重
    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        index_weight = 0
        for i in range(len(self.output_layer.neurons)):
            for j in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[i].weights.append(random.random)
                else:
                    self.output_layer.neurons[i].weights.append(output_layer_weights[index_weight])
                index_weight += 1

    # 打印整个网络的结构
    def inspect(self):
        print('-----------')
        print(' *Inputs: {}'.format(self.num_inputs))
        print('-----------')
        print(' *Hidden layer')
        self.hidden_layer.inspect()
        print('-----------')
        print(' *Output layer')
        self.output_layer.inspect()
        print('-----------')

    # 前向传播
    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def calculate_total_error(self, outputs):
        total_error = 0
        for i in range(len(self.output_layer.neurons)):
            total_error += self.output_layer.neurons[i].output - outputs[i]
        return total_error

    # 反向传播训练
    def back_propagation_train(self, inputs, target_outputs):
        # 前向传播一次，得到结果
        self.feed_forward(inputs)
        error = []

        for i in range(len(self.output_layer.neurons)):
            error.append(self.output_layer.neurons[i].calculate_error(target_outputs[i]))

        # print("output error is {}".format(error))

        pd_output_layer_out = []

        # 得到输出层单个神经元对整体输入的偏导
        for i in range(len(self.output_layer.neurons)):
            pd_output_layer_out.append(self.output_layer.neurons[i].calculate_pd_error_out(target_outputs[i]) * self.output_layer.neurons[i].calculate_pd_error_sigmod())

        pd_output_layer_weights = []
        index = 0

        # 得到输出层单个神经元对各个输入权值的偏导
        for i in range(len(self.output_layer.neurons)):
            for j in range(len(self.hidden_layer.neurons)):
                pd_output_layer_weights.append(pd_output_layer_out[i] * self.hidden_layer.outputs[index])
                index += index

        # print("pd for output layer weight is {}".format(pd_output_layer_weights))

        pd_output_layer_to_hidden_layer_outputs = [0] * len(self.hidden_layer.neurons)

        # 得到输出层单个神经元对隐藏层各个输出的偏导之和
        for i in range(len(self.output_layer.neurons)):
            for j in range(len(self.hidden_layer.neurons)):
                pd_output_layer_to_hidden_layer_outputs[j] += pd_output_layer_out[i] * self.output_layer.neurons[i].weights[j]

        pd_hidden_layer_weights = []

        # 得到隐藏层各个神经元对各个输入权值的偏导
        for i in range(len(self.hidden_layer.neurons)):
            for j in range(self.num_inputs):
                pd_hidden_layer_weights.append(pd_output_layer_to_hidden_layer_outputs[i] * self.hidden_layer.neurons[i].calculate_pd_error_sigmod() * inputs[j])

        # print("pd for hidden layer weight is {}".format(pd_hidden_layer_weights))

        index = 0
        # 更新输出层权值
        for i in range(len(self.output_layer.neurons)):
            for j in range(len(self.output_layer.neurons[i].weights)):
                self.output_layer.neurons[i].weights[j] -= self.LEARNING_RATE * pd_output_layer_weights[index]
                index += 1

        index = 0
        # 更新隐藏层权值
        for i in range(len(self.hidden_layer.neurons)):
            for j in range(len(self.hidden_layer.neurons[i].weights)):
                self.hidden_layer.neurons[i].weights[j] -= self.LEARNING_RATE * pd_hidden_layer_weights[index]
                index += 1

nn = NeuralNetwork(2, 3, 1, hidden_layer_weights=[0.15, 0.2, 0.1, 0.25, 0.3, 0.4], hidden_layer_bias=0.3, output_layer_weights=[0.4, 0.45, 0.5], output_layer_bias=0.3)
inputs = [0, 1]
target_outputs = [1]
train_rounds = 1000
x = np.linspace(0, train_rounds - 1, train_rounds)
y = []
for j in range(train_rounds):
    nn.back_propagation_train(inputs, target_outputs)
    error = nn.calculate_total_error(target_outputs)
    print("total error is {}".format(error))
    y.append(error)

y = np.array(y)

plt.plot(x, y, label="$error$", color="red", linewidth=2)
plt.xlabel("Time(s)")
plt.ylabel("Error")
plt.title("Error variation")
plt.ylim(-0.5, 1.5)
plt.legend()
plt.show()

print('Final output is {}'.format(nn.feed_forward(inputs)))
