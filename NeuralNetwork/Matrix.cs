using System;

namespace Salty
{
    public static partial class Maths
    { 
        /// <summary>
        /// Represents a matrix object with float entries. This is a collection of floats, arranged in rows and 
        /// columns.
        /// </summary>
        public class Matrix
        {
            /// <summary>
            /// The number of rows in the current matrix.
            /// </summary>
            public readonly int RowCount;

            /// <summary>
            /// The number of columns in the current matrix.
            /// </summary>
            public readonly int ColumnCount;

            /// <summary>
            /// The 2-dimensional array that holds all entries for the current matrix.
            /// </summary>
            private readonly float[,] data;

            /// <summary>
            /// Creates a new matrix with all entries 0.0f.
            /// </summary>
            /// <param name="rowCount">The number of rows in the current matrix. Must be greater than zero.</param>
            /// <param name="columnCount">The number of columns in the current matrix. Must be greater than zero.
            /// </param>
            public Matrix(int rowCount, int columnCount)
            {
                if (rowCount <= 0 || columnCount <= 0)
                {
                    throw new ArgumentOutOfRangeException("matrix must have positive number of rows and columns");
                }
                RowCount = rowCount;
                ColumnCount = columnCount;
                this.data = new float[RowCount, ColumnCount];
            }

            public Matrix(float[,] data)
            {
                RowCount = data.GetLength(0);
                ColumnCount = data.GetLength(1);
                this.data = data;
            }

            /// <summary>
            /// Creates a new zero matrix.
            /// </summary>
            /// <param name="rowCount">The number of rows in the zero matrix. Must be greater than zero.</param>
            /// <param name="columnCount">The number of columns in the zero matrix. Must be greater than zero.
            /// </param>
            /// <returns></returns>
            public static Matrix Zero(int rowCount, int columnCount)
            {
                return new Matrix(rowCount, columnCount);
            }

            /// <summary>
            /// Creates a new one matrix. That is, a matrix with all entries being equal to 1.
            /// </summary>
            /// <param name="rowCount">The number of rows in the one matrix. Must be greater than zero.</param>
            /// <param name="columnCount">The number of columns in the one matrix. Must be greater than zero.
            /// </param>
            /// <returns></returns>
            public static Matrix One(int rowCount, int columnCount)
            {
                Matrix one = new Matrix(rowCount, columnCount);
                one.Apply(x => 1f);
                return one;
            }

            /// <summary>
            /// Creates a new unit matrix.
            /// </summary>
            /// <param name="size">The number of rows and columns in the unit matrix. Must be greater than zero.
            /// </param>
            /// <returns></returns>
            public static Matrix Unit(int size)
            {
                Matrix unit = Zero(size, size);
                for (int i = 0; i < size; i++)
                {
                    unit[i, i] = 1.0f;
                }
                return unit;
            }

            /// <summary>
            /// Creates a new matrix with all random entries between -1 (inclusive) and 1 (exclusive).
            /// </summary>
            /// <param name="rowCount">The number of rows in the random matrix. Must be greater than zero.</param>
            /// <param name="columnCount">The number of columns in the random matrix. Must be greater than zero.
            /// </param>
            /// <returns>The newly created random matrix.</returns>
            public static Matrix Random(int rowCount, int columnCount, Random rng)
            {
                Matrix random = new Matrix(rowCount, columnCount);
                random.Randomise(rng);
                return random;
            }

            /// <summary>
            /// Creates a new matrix with one column with values from the given array.
            /// </summary>
            /// <param name="column">The array whose values will fill this column matrix.</param>
            /// <returns>The newly created column matrix.</returns>
            public static Matrix FromArray(float[] column) 
            {
                Matrix matrix = new Matrix(column.Length, 1);
                matrix.SetColumn(0, column);
                return matrix;
            }

            /// Adds two matrices of the same dimensions together.
            /// </summary>
            /// <param name="a">A matrix to be added.</param>
            /// <param name="b">A matrix to be added.</param>
            /// <returns>The matrix sum of the two matrices to be added.</returns>
            public static Matrix operator +(Matrix a, Matrix b)
            {
                Matrix aCopy = a.Copy();
                aCopy.Add(b);
                return aCopy;
            }

            /// <summary>
            /// Subtracts one matrix from another of the same dimensions.
            /// </summary>
            /// <param name="a">The left matrix.</param>
            /// <param name="b">The right matrix which will be subtracted.</param>
            /// <returns>The difference of the two matrices.</returns>
            public static Matrix operator -(Matrix a, Matrix b)
            {
                return a + (-b);
            }

            public static Matrix Hadamard(Matrix a, Matrix b)
            {
                if (a.RowCount != b.RowCount || a.ColumnCount != b.ColumnCount)
                {
                    throw new ArgumentException("matrices must have the same dimensions for Hadamard product");
                }
                Matrix hadamard = new Matrix(a.RowCount, a.ColumnCount);
                for (int row = 0; row < a.RowCount; row++)
                {
                    for (int col = 0; col < a.ColumnCount; col++)
                    {
                        hadamard[row, col] = a[row, col] * b[row, col];
                    }
                }

                return hadamard;
            }

            /// <summary>
            /// Multiplies two matrices together. This only works when the number of columns in the left matrix 
            /// equals the number of rows in the right matrix.
            /// </summary>
            /// <param name="a">The left matrix to be multiplied.</param>
            /// <param name="b">The right matrix to be multiplied.</param>
            /// <returns>The matrix product of the two matrices to be multiplied.</returns>
            public static Matrix operator *(Matrix a, Matrix b)
            {
                if (a.ColumnCount != b.RowCount)
                {
                    throw new ArgumentException("number of columns in left matrix must be equal to number of " +
                        "rows in right matrix");
                }

                // a.ColumnCount equals b.RowCount so just call this n
                int n = a.ColumnCount;

                Matrix product = Zero(a.RowCount, b.ColumnCount);

                for (int row = 0; row < product.RowCount; row++)
                {
                    float[] aRow = new float[n];

                    for (int col = 0; col < product.ColumnCount; col++)
                    {
                        float[] bCol = new float[n];

                        for (int i = 0; i < n; i++)
                        {
                            aRow[i] = a[row, i];
                            bCol[i] = b[i, col];
                        }

                        // calculate the dot product of aRow and bCol and store it in this entry
                        for (int i = 0; i < n; i++)
                        {
                            product[row, col] += aRow[i] * bCol[i];
                        }
                    }
                }

                return product;
            }

            /// <summary>
            /// Returns a scaled copy of a matrix. This does not change the matrix.
            /// </summary>
            /// <param name="k">The scalar.</param>
            /// <param name="m">The matrix to get the scaled copy of.</param>
            /// <returns>A scaled copy of the matrix.</returns>
            public static Matrix operator *(float k, Matrix m)
            {
                Matrix copy = m.Copy();
                copy.Scale(k);
                return copy;
            }

            /// <summary>
            /// Returns a scaled copy of a matrix. This does not change the matrix.
            /// </summary>
            /// <param name="m">The matrix to get the scaled copy of.</param>
            /// <param name="k">The scalar.</param>
            /// <returns>A scaled copy of the matrix.</returns>
            public static Matrix operator *(Matrix m, float k)
            {
                return k * m;
            }

            /// <summary>
            /// Returns the negative of a matrix. This does not change the matrix.
            /// </summary>
            /// <param name="m">The matrix to get the negative of.</param>
            /// <returns>The negative of the matrix.</returns>
            public static Matrix operator -(Matrix m)
            {
                Matrix copy = m.Copy();
                return -1 * m;
            }

            /// <summary>
            /// Determines whether two matrices are equal or not. That is, if each matrix has element-wise equal 
            /// float entries.
            /// </summary>
            /// <param name="a">A matrix to test equality.</param>
            /// <param name="b">A matrix to test equality.</param>
            /// <returns>True if and only if both matrices are equal.</returns>
            public static bool operator ==(Matrix a, Matrix b)
            {
                return a.Equals(b);
            }

            /// <summary>
            /// Determines whether two matrices are unequal or not.
            /// </summary>
            /// <param name="a">A matrix to test inequality.</param>
            /// <param name="b">A matrix to test inequality.</param>
            /// <returns>True if and only if both matrices are unequal.</returns>
            public static bool operator !=(Matrix a, Matrix b)
            {
                return !a.Equals(b);
            }

            /// <summary>
            /// Sets (overwrites) the value of a particular entry in the current matrix.
            /// </summary>
            /// <param name="row">The row of the entry to set.</param>
            /// <param name="col">The column of the entry to set.</param>
            /// <param name="value">The desired value of the entry to set.</param>
            /// <returns>The old value of the entry before the new value was set.</returns>
            public float Set(int row, int col, float value)
            {
                float oldValue = value;
                HandleIndexException(row, col);
                this.data[row, col] = value;
                return oldValue;
            }

            /// <summary>
            /// Sets the values of a row in the current matrix to the values of an array.
            /// </summary>
            /// <param name="row">The index of the row to set.</param>
            /// <param name="values">The values to set this row to. Must have length equal to the number of columns 
            /// in this matrix.</param>
            /// <returns>The old row before the new values were set.</returns>
            public float[] SetRow(int row, float[] values)
            {
                float[] oldRow = GetRow(row);
                if (values.Length != this.ColumnCount)
                {
                    throw new ArgumentException("values to set row must have length equal to the number of " +
                        "columns in the matrix");
                }
                for (int i = 0; i < values.Length; i++)
                {
                    this[row, i] = values[i];
                }
                return oldRow;
            }

            /// <summary>
            /// Sets the values of a column in the current matrix to the values of an array.
            /// </summary>
            /// <param name="col">The index of the column to set.</param>
            /// <param name="values">The values to set this column to. Must have length equal to the number of rows in the 
            /// curent matrix.</param>
            /// <returns>The old column before the new values were set.</returns>
            public float[] SetColumn(int col, float[] values)
            {
                float[] oldColumn = GetColumn(col);
                if (values.Length != this.RowCount)
                {
                    throw new ArgumentException("values to set column must have length equal to the number of " +
                        "rows in the matrix");
                }
                for (int i = 0; i < values.Length; i++)
                {
                    this[i, col] = values[i];
                }
                return oldColumn;
            }

            /// <summary>
            /// Returns the value of a particular entry in the current matrix.
            /// </summary>
            /// <param name="row">The row of the entry to set.</param>
            /// <param name="col">The column of the entry to get.</param>
            /// <returns>The value of the entry in the current matrix at the specified row and column.</returns>
            public float Get(int row, int col)
            {
                HandleIndexException(row, col);
                return this.data[row, col];
            }

            /// <summary>
            /// Returns a row of the current matrix as an array of its entries.
            /// </summary>
            /// <param name="row">The row to get.</param>
            /// <returns>An array containing the entries of the desired row, where higher indexed elements of the 
            /// array represent entries of the matrix row which are further to the right.</returns>
            public float[] GetRow(int row)
            {
                float[] values = new float[ColumnCount];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = this[row, i];
                }
                return values;
            }

            /// <summary>
            /// Returns a column of the current matrix as an array of its entries.
            /// </summary>
            /// <param name="col">The column to get.</param>
            /// <returns>An array containing the entries of the desired column, where higher indexed elements of the 
            /// array represent entries of the matrix column which are further down.</returns>
            public float[] GetColumn(int col)
            {
                float[] values = new float[RowCount];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = this[i, col];
                }
                return values;
            }

            /// <summary>
            /// Overloads the [] operator to get and set values in the current matrix.
            /// </summary>
            /// <param name="row">The row to index.</param>
            /// <param name="col">The column to index.</param>
            public float this[int row, int col]
            {
                get
                {
                    return Get(row, col);
                }
                set
                {
                    Set(row, col, value);
                }
            }

            /// <summary>
            /// Overloads the [] operator to get and set rows in the current matrix.
            /// </summary>
            /// <param name="row">The row to index.</param>
            public float[] this[int row]
            {
                get
                {
                    return GetRow(row);
                }
                set
                {
                    SetRow(row, value);
                }
            }

            /// <summary>
            /// Fills the current matrix with random entries between -1 (inclusive) and 1 (exclusive).
            /// </summary>
            public void Randomise(Random rng)
            {
                for (int row = 0; row < RowCount; row++)
                {
                    for (int col = 0; col < ColumnCount; col++)
                    {
                        this[row, col] = 2.0f * (float)rng.NextDouble() - 1.0f;
                    }
                }
            }

            /// <summary>
            /// Fills the current matrix with random entries between -1 (inclusive) and 1 (exclusive).
            /// </summary>
            public void Randomize(Random rng)
            {
                Randomise(rng);
            }

            /// <summary>
            /// Adds another matrix to the current matrix. The other matrix must be of equal dimensions to this one.
            /// </summary>
            /// <param name="other">The other matrix to add to this one. Must be of equal dimensions to the current 
            /// matrix.</param>
            public void Add(Matrix other)
            {
                if (this.RowCount != other.RowCount || this.ColumnCount != other.ColumnCount)
                {
                    throw new ArgumentException("summand matrices must be of equal dimension");
                }

                // this and other have the same dimensions, so we may just use this.RowCount and this.ColumnCount
                for (int row = 0; row < RowCount; row++)
                {
                    for (int col = 0; col < ColumnCount; col++)
                    {
                        this[row, col] += other[row, col];
                    }
                }
            }

            /// <summary>
            /// Multiplies the current matrix by a scalar.
            /// </summary>
            /// <param name="k">The scalar to multiply the current matrix by.</param>
            public void Scale(float k)
            {
                for (int row = 0; row < RowCount; row++)
                {
                    for (int col = 0; col < ColumnCount; col++)
                    {
                        this[row, col] *= k;
                    }
                }
            }

            /// <summary>
            /// Sets the value of all entries in the current matrix.
            /// </summary>
            /// <param name="n">The value to assign all entries in the current matrix to.</param>
            public void Fill(float n)
            {
                for (int row = 0; row < RowCount; row++)
                {
                    for (int col = 0; col < ColumnCount; col++)
                    {
                        this[row, col] = n;
                    }
                }
            }

            /// <summary>
            /// Returns the transpose of the current matrix. Does not change the current matrix.
            /// </summary>
            /// <returns>The transpose of the current matrix. That is, a copy of the current matrix with rows and 
            /// columns swapped.</returns>
            public Matrix Transpose()
            {
                Matrix transpose = new Matrix(ColumnCount, RowCount);

                for (int row = 0; row < RowCount; row++)
                {
                    for (int col = 0; col < ColumnCount; col++)
                    {
                        transpose[col, row] = this[row, col];
                    }
                }

                return transpose;
            }

            /// <summary>
            /// Applies a function to every element of the current matrix (element-wise).
            /// </summary>
            /// <param name="function">The function to take each element of the current matrix as inputs and update 
            /// those elements with its return value.</param>
            public void Apply(Func<float, float> function)
            {
                for (int row = 0; row < RowCount; row++)
                {
                    for (int col = 0; col < ColumnCount; col++)
                    {
                        this[row, col] = function(this[row, col]);
                    }
                }
            }

            /// <summary>
            /// Returns a matrix whose elements are the return value of the given function acting element-wise on 
            /// the given matrix. Does not change the given matrix.
            /// </summary>
            /// <param name="m">The matrix whose elements will be used in determining the result.</param>
            /// <param name="function">The function whose return values will determine the resulting matrix's 
            /// elements.</param>
            /// <returns>A matrix whose elements are the return value of the given function acting element-wise on 
            /// the given matrix.</returns>
            public static Matrix Apply(Matrix m, Func<float, float> function)
            {
                Matrix copy = m.Copy();
                copy.Apply(function);
                return copy;
            }

            /// <summary>
            /// Performs a deep copy of the current matrix and returns the result.
            /// </summary>
            /// <returns>A deep copy of the current matrix.</returns>
            public Matrix Copy()
            {
                Matrix copy = new Matrix(RowCount, ColumnCount);
                for (int row = 0; row < copy.RowCount; row++)
                {
                    for (int col = 0; col < copy.ColumnCount; col++)
                    {
                        copy[row, col] = this[row, col];
                    }
                }
                return copy;
            }

            /// <summary>
            /// Prints the current matrix to the console in a tabular view.
            /// </summary>
            public void Print()
            {
                Console.WriteLine(this.ToString());
            }

            /// <summary>
            /// Prints the given matrix to the console in a tabular view.
            /// </summary>
            /// <param name="m">The matrix to print.</param>
            public static void Print(Matrix m)
            {
                m.Print();
            }


            /// <summary>
            /// Returns a <see cref="System.String" /> that represents the current matrix.
            /// </summary>
            /// <returns>A <see cref="System.String" /> that represents the current matrix.</returns>
            public override string ToString()
            {
                string repr = "";
                for (int row = 0; row < RowCount; row++)
                {
                    for (int col = 0; col < ColumnCount; col++)
                    {
                        repr += this[row, col] + "\t";
                    }
                    repr += "\n";
                }
                return repr;
            }



            /// <summary>
            /// Determines whether a matrix is equal to this one or not. That is, if each matrix has element-wise 
            /// equal float entries.
            /// </summary>
            /// <param name="obj">A matrix to test equality.</param>
            /// <returns>True if and only if both matrices are equal.</returns>
            public override bool Equals(object obj)
            {
                if (obj == null || !(obj is Matrix))
                {
                    return false;
                }

                Matrix other = (Matrix)obj;

                if (this.RowCount != other.RowCount || this.ColumnCount != other.ColumnCount)
                {
                    return false;
                }

                /* the current matrix and the other matrix have the same dimensions, so generically call this 
                 * RowCount and cols. */
                for (int row = 0; row < RowCount; row++)
                {
                    for (int col = 0; col < ColumnCount; col++)
                    {
                        if (other[row, col] != this[row, col])
                        {
                            return false;
                        }
                    }
                }

                return true;
            }

            /// <summary>
            /// Returns the hashcode for the current matrix.
            /// </summary>
            /// <returns>The hashcode for the current matrix.</returns>
            public override int GetHashCode()
            {
                int hashCode = 23;

                // Don't check for overflow on integer arithmetic - just wrap.
                unchecked
                {
                    for (int row = 0; row < RowCount; row++)
                    {
                        for (int col = 0; col < ColumnCount; col++)
                        {
                            hashCode = hashCode * 31 + this[row, col].GetHashCode();
                        }
                    }
                }

                return hashCode;
            }

            /// <summary>
            /// Private helper function. ThRowCount an exception if the position at the specified row and column in 
            /// the current matrix is invalid. 
            /// </summary>
            /// <param name="row">The row to check validity.</param>
            /// <param name="col">The column to check validity.</param>
            private void HandleIndexException(int row, int col)
            {
                if (row < 0 || row >= RowCount || col < 0 || col >= ColumnCount)
                {
                    throw new ArgumentOutOfRangeException("position in matrix must be within matrix dimensions");
                }
            }
        }
    }
}
