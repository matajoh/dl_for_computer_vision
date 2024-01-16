from typing import List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

Step = NamedTuple("Step", [("x", float), ("y", float), ("dx", float)])


def objective(x):
    return x*x - 5*x + 2


def dobjective(x):
    return 2*x - 5


def optimize(x, eta, num_steps, momentum=0.0):
    steps = []
    prev_grad = 0.0
    for _ in range(num_steps):
        dx = dobjective(x)
        dx = dx + momentum*prev_grad
        steps.append(Step(x, objective(x), eta*dx))
        x -= eta*dx
        prev_grad = dx

    return steps


def plot_step(ax, step, marker="bo", grad_color="b", label="dx"):
    x_values = np.arange(-30, 80) / 10.0
    y_values = objective(x_values)
    ax.set_xlim(-3, 8)
    ax.plot(x_values, y_values, c="black")
    ax.plot(step.x, step.y, marker, label=label, ms=15, zorder=4)
    if abs(step.dx) > 0.25:
        ax.arrow(step.x, step.y, -step.dx, 0, head_width=1.5, head_length=.25,
                 length_includes_head=True, fc=grad_color, ec=grad_color)


def gradient_descent():
    steps = optimize(-2, 0.65, 4)
    plt.rc('font', size=15)
    plt.figure(figsize=(8, 6))
    for i, step in enumerate(steps):
        ax = plt.subplot(2, 2, i+1)
        ax.set_title(str(i))
        plot_step(ax, step)

    plt.tight_layout()
    plt.show()


def momentum():
    steps = optimize(-2, 0.15, 4)
    steps_m = optimize(7, 0.15, 4, momentum=0.4)
    plt.rc('font', size=15)
    plt.figure(figsize=(8, 6))
    for i, (step, step_m) in enumerate(zip(steps, steps_m)):
        ax = plt.subplot(2, 2, i+1)
        ax.set_title(str(i))
        plot_step(ax, step)
        plot_step(ax, step_m, "g^", "g", "dx+mom")
        ax.legend(loc='upper center')

    plt.tight_layout()
    plt.show()


def main():
    gradient_descent()
    momentum()


if __name__ == "__main__":
    main()
