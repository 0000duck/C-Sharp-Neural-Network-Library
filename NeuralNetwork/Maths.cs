namespace Salty
{
    /// <summary>A static class for maths-related functions.</summary>
    public static partial class Maths
    {
        /// <summary>Returns the sum of all elements in an array.</summary>
        /// <param name="nums">The floating point numbers to make up the 
        /// elements of the array to sum.</param>
        /// <returns>The sum of the all elements in the array.</returns>
        public static float Sum(params float[] nums)
        {
            float sum = 0f;
            foreach (float num in nums)
            {
                sum += num;
            }
            return sum;
        }
    }
}
