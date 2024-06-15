from typing import Any


def sign(x):
    """Return the sign of a number"""
    return int(x / abs(x)) if x != 0 else 1


class FFraction:
    """Class for working with fractions"""

    def __init__(self, numerator, denominator) -> None:
        if denominator == 0:
            raise ZeroDivisionError("Cannot have a zero denominator")

        self.numerator = numerator
        self.denominator = denominator
        self.simplify()

    def evaluate(self):
        """Return the fraction's evaluation"""
        return self.numerator / self.denominator

    def simplify(self):
        """Simplify the fraction"""
        gcd = FFraction.get_gcd(self.numerator, self.denominator)
        self.numerator //= gcd
        self.denominator //= gcd

        if self.denominator < 0:
            self.numerator *= -1
            self.denominator *= -1

    @staticmethod
    def get_gcd(a, b):
        """Returns the greatest common divisor of two numbers using the Euclidean algorithm"""
        while b != 0:
            a, b = b, a % b
        return a

    def __add__(self, other):
        """Add two fractions"""
        if isinstance(other, FFraction):
            other = FFraction(other, 1)
            result = FFraction(self.numerator * other.denominator +
                            self.denominator * other.numerator,
                            self.denominator * other.denominator)
            result.simplify()
            return result
        else:
            return self.evaluate() + other

    def __sub__(self, other):
        """Subtract two fractions"""
        if isinstance(other, FFraction):
            other = FFraction(other, 1)
            result = FFraction(self.numerator * other.denominator -
                              self.denominator * other.numerator,
                              self.denominator * other.denominator)
            result.simplify()
            return result
        else:
            return self.evaluate() - other

    def __mul__(self, other):
        """Multiply two fractions"""
        if isinstance(other, FFraction):
            other = FFraction(other, 1)
            result = FFraction(self.numerator * other.numerator,
                              self.denominator * other.denominator)
            result.simplify()
            return result
        else:
            return self.evaluate() * other

    def __truediv__(self, other):
        """Divide two fractions"""
        if isinstance(other, FFraction):
            other = FFraction(other, 1)
            result = FFraction(self.numerator * other.denominator,
                              self.denominator * other.numerator)
            result.simplify()
            return result
        else:
            return self.evaluate() / other

    def __repr__(self) -> str:
        """Return the string representation of the fraction"""
        return f"({self.numerator}/{self.denominator})" if self.evaluate() != 0 else "0"

class Rational:
    """Class for working with fractions"""

    def __init__(self, numerator, denominator) -> None:
        if denominator == 0:
            raise ZeroDivisionError("Cannot have a zero denominator")

        self.numerator = numerator
        self.denominator = denominator

    def evaluate(self):
        """Return the fraction's evaluation"""
        return self.numerator() / self.denominator()

    def simplify(self):
        """Simplify the fraction"""
        gcd = FFraction.get_gcd(self.numerator, self.denominator)
        self.numerator //= gcd
        self.denominator //= gcd

        if self.denominator < 0:
            self.numerator *= -1
            self.denominator *= -1

    @staticmethod
    def get_gcd(a, b):
        """Returns the greatest common divisor of two numbers using the Euclidean algorithm"""
        while b != 0:
            a, b = b, a % b
        return a

    def __add__(self, other):
        """Add two fractions"""
        if isinstance(other, FFraction):
            other = FFraction(other, 1)
            result = FFraction(self.numerator * other.denominator +
                            self.denominator * other.numerator,
                            self.denominator * other.denominator)
            return result
        else:
            return self.evaluate() + other

    def __sub__(self, other):
        """Subtract two fractions"""
        if isinstance(other, FFraction):
            other = FFraction(other, 1)
            result = FFraction(self.numerator * other.denominator -
                              self.denominator * other.numerator,
                              self.denominator * other.denominator)

            return result
        else:
            return self.evaluate() - other

    def __mul__(self, other):
        """Multiply two fractions"""
        if isinstance(other, FFraction):
            other = FFraction(other, 1)
            result = FFraction(self.numerator * other.numerator,
                              self.denominator * other.denominator)
            return result
        else:
            return self.evaluate() * other

    def __truediv__(self, other):
        """Divide two fractions"""
        if isinstance(other, FFraction):
            other = FFraction(other, 1)
            result = FFraction(self.numerator * other.denominator,
                              self.denominator * other.numerator)
            return result
        else:
            return self.evaluate() / other

    def __repr__(self) -> str:
        """Return the string representation of the fraction"""
        return f"Fraction({self.numerator}/{self.denominator})" if self.evaluate() != 0 else "0"

    def __str__1(self) -> str: # Potential way to represent fractions
        """Return the string representation of the fraction"""
        n_str = str(self.numerator)
        d_str = str(self.denominator)
        return (n_str + '\n' + '-' * max(len(n_str), len(d_str)) + '\n' + d_str) if self.evaluate() != 0 else "0"
        
    def __call__(self, *args, **kwargs) -> Any:
        return self.evaluate()