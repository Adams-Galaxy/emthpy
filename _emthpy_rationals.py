

def sign(x):
    """Return the sign of a number"""
    return int(x / abs(x)) if x != 0 else 1

class Rational:
    """(WIP) Class for working with fractions"""

    def __init__(self, numerator, denominator) -> None:
        if denominator == 0:
            raise ZeroDivisionError("Cannot have a zero denominator")

        self.numerator = numerator
        self.denominator = denominator

    def evaluate(self):
        """Return the fraction's evaluation"""
        result = self.numerator / self.denominator
        return result if result // 1 != result else int(result)

    def floor_evaluate(self):
        """Return the floor of the fraction's evaluation"""
        return self.evaluate() // 1

    def simplify(self):
        """Simplify the fraction"""
        Rational.rat_simplify(self)

    def try_simplify(self):
        """Attempt to simplify the fraction"""
        try:
            Rational.rat_simplify(self)
            return True
        except ZeroDivisionError:
            return False
        except TypeError:
            return False


    @staticmethod
    def get_gcd(a, b):
        """Returns the greatest common divisor of two numbers using the Euclidean algorithm"""
        while b != 0:
            a, b = b, a % b
        return a

    @staticmethod
    def rat_simplify(rational):
        """Simplify a fraction"""
        if not isinstance(rational.numerator, (int, float)) and isinstance(rational.denominator, (int, float)):
            raise TypeError("Both the numerator and the denominator must be integers or floats")

        gcd = Rational.get_gcd(rational.numerator, rational.denominator)
        rational.numerator //= gcd
        rational.denominator //= gcd

        if rational.denominator < 0:
            rational.numerator *= -1
            rational.denominator *= -1

    def __add__(self, other):
        """Add two fractions"""
        if not isinstance(other, Rational):
            other = Rational(other, 1)
        result = Rational(self.numerator * other.denominator +
                            self.denominator * other.numerator,
                            self.denominator * other.denominator)
        result.try_simplify()
        return result
    def __radd__(self, other):
        """Add a fraction and an integer"""
        if not isinstance(other, (Rational, int, float)):
            return other + self.evaluate()
        return self.__add__(other)
    def __sub__(self, other):
        """Subtract two fractions"""
        if not isinstance(other, Rational):
            other = Rational(other, 1)
        result = Rational(self.numerator * other.denominator -
                            self.denominator * other.numerator,
                            self.denominator * other.denominator)
        result.try_simplify()
        return result
    def __rsub__(self, other):
        """Subtract a fraction from an integer"""
        if not isinstance(other, (Rational, int, float)):
            return other - self.evaluate()
        return self.__sub__(other)
    def __mul__(self, other):
        """Multiply two fractions"""
        if not isinstance(other, Rational):
            other = Rational(other, 1)
        result = Rational(self.numerator * other.numerator,
                            self.denominator * other.denominator)
        result.try_simplify()
        return result
    def __rmul__(self, other):
        """Multiply a fraction and an integer"""
        if not isinstance(other, (Rational, int, float)):
            return other * self.evaluate()
        return self.__mul__(other)
    def __truediv__(self, other):
        """Divide two fractions"""
        if not isinstance(other, Rational):
            other = Rational(other, 1)
        result = Rational(self.numerator * other.denominator,
                            self.denominator * other.numerator)
        result.try_simplify()
        return result
    def __rtruediv__(self, other):
        """Divide an integer by a fraction"""
        if not isinstance(other, (Rational, int, float)):
            return other / self.evaluate()
        return self.__truediv__(other)
    def __neg__(self):
        """Negate the fraction"""
        return Rational(-self.numerator, self.denominator)

    def __eq__(self, other):
        """Check if two fractions are equal"""
        if isinstance(other, Rational):
            return self.numerator == other.numerator and self.denominator == other.denominator
        return other == self.evaluate()

    def __repr__(self) -> str:
        """Return the string representation of the fraction"""
        return f"Rational({self.numerator}/{self.denominator})"
    def __str__(self) -> str:
        """Return the string representation of the fraction"""
        if not isinstance(self.numerator, (int, float)) and isinstance(self.denominator, (int, float)):
            return f"{self.numerator}/{self.denominator}"
        if self.denominator == 1:
            return str(self.numerator)
        floored = self.floor_evaluate()
        if self.evaluate() == floored:
            return str(floored)
        return f"{self.numerator}/{self.denominator}"

    def __str__1(self) -> str: # Potential way to represent fractions
        """Return the string representation of the fraction"""
        n_str = str(self.numerator)
        d_str = str(self.denominator)
        return (n_str + '\n' + '-' * max(len(n_str), len(d_str)) + '\n' + d_str) if self.evaluate() != 0 else "0"
        
    def __call__(self, *args, **kwargs):
        return self.evaluate()