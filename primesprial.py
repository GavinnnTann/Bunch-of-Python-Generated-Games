import numpy as np
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    import matplotlib.pyplot as plt

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0:
                return False
        return True

    def spiral_positions(n_max):
        # yields (num, x, y, is_prime)
        x = y = 0
        dx, dy = 0, -1
        for num in range(1, n_max + 1):
            yield num, x, y, is_prime(num)
            # turn logic to create an outward square spiral
            if (x == y) or (x > 0 and x == 1 - y) or (x < 0 and x == -y):
                dx, dy = -dy, dx
            x, y = x + dx, y + dy

    def animate_spiral(grid_size=201, interval=1):
        n_max = grid_size**2
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Prime Number Spiral (growing)')
        ax.grid(True, linewidth=0.4, linestyle='--', color='lightgray')
        ax.axhline(0, color='gray', linewidth=0.6)
        ax.axvline(0, color='gray', linewidth=0.6)

        scatter = ax.scatter([], [], s=6, c='black')
        txt = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top')

        primes_x = []
        primes_y = []
        count_primes = 0

        # Pre-generate positions to iterate quickly in the animation function
        positions = list(spiral_positions(n_max))

        def init():
            ax.set_xlim(-grid_size//2 - 1, grid_size//2 + 1)
            ax.set_ylim(-grid_size//2 - 1, grid_size//2 + 1)
            scatter.set_offsets(np.empty((0, 2)))
            txt.set_text('')
            return scatter, txt

        def update(frame):
            nonlocal count_primes
            num, x, y, prime = positions[frame]
            if prime:
                primes_x.append(x)
                primes_y.append(y)
                count_primes += 1
            if primes_x:
                offsets = np.column_stack((primes_x, primes_y))
                scatter.set_offsets(offsets)
            # autoscale to fit new points while keeping square aspect
            min_x = min(primes_x) if primes_x else -grid_size//2
            max_x = max(primes_x) if primes_x else grid_size//2
            min_y = min(primes_y) if primes_y else -grid_size//2
            max_y = max(primes_y) if primes_y else grid_size//2
            margin = 1
            ax.set_xlim(min_x - margin, max_x + margin)
            ax.set_ylim(min_y - margin, max_y + margin)
            txt.set_text(f'n={num}  primes={count_primes}')
            return scatter, txt

        ani = FuncAnimation(fig, update, frames=len(positions), init_func=init,
                            blit=False, interval=interval, repeat=False)
        plt.show()

    if __name__ == '__main__':
        # Change grid_size to increase/decrease how many numbers are generated (grid_size**2 numbers)
        animate_spiral(grid_size=101, interval=2)
    return True

def prime_spiral(size):
    spiral = np.zeros((size, size), dtype=int)
    x, y = size // 2, size // 2
    dx, dy = 0, -1
    num = 1
    for _ in range(size**2):
        if 0 <= x < size and 0 <= y < size:
            spiral[y, x] = is_prime(num)
        if (x == y) or (x < y and x + y == size - 1) or (x > y and x + y == size):
            dx, dy = -dy, dx
        x, y = x + dx, y + dy
        num += 1
    return spiral

size = 101  # Change size for larger/smaller spirals
spiral = prime_spiral(size)

plt.figure(figsize=(8,8))
plt.imshow(spiral, cmap='Greys', interpolation='nearest')
plt.axis('off')
plt.title('Prime Number Spiral')
plt.show()