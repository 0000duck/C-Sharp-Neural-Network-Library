using System;
using System.Collections.Generic;
using System.Text;

namespace Salty
{
    public static partial class Maths
    {
        public static float[] Apply(float[] nums, Func<float, float> function)
        {
            float[] appliedNums = { };
            for (int i = 0; i < nums.Length; i++)
            {
                appliedNums[i] = function(nums[i]);
            }
            return appliedNums;
        }

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
