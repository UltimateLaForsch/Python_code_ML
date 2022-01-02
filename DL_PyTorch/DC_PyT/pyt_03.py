import torch
from Net import Net

input_layer = torch.tensor([0.1055, 0.7335, 0.3182, 0.2274, 0.0118, 0.1372, 0.5870, 0.7781, 0.2103,
        0.4639, 0.8561, 0.8264, 0.2890, 0.0212, 0.8563, 0.3475, 0.0667, 0.6863,
        0.5355, 0.5144, 0.0424, 0.9384, 0.1137, 0.2406, 0.8575, 0.6936, 0.7456,
        0.8508, 0.3865, 0.4262, 0.4654, 0.5171, 0.8495, 0.6024, 0.0645, 0.3962,
        0.1978, 0.3831, 0.0145, 0.8251, 0.0854, 0.6428, 0.5116, 0.1053, 0.7481,
        0.4332, 0.3441, 0.4826, 0.1331, 0.2671, 0.7410, 0.5192, 0.0659, 0.1684,
        0.0292, 0.2557, 0.2280, 0.7159, 0.2078, 0.6814, 0.4815, 0.5788, 0.0051,
        0.0575, 0.5968, 0.4848, 0.3814, 0.0415, 0.5373, 0.4863, 0.4655, 0.3593,
        0.1093, 0.9697, 0.4330, 0.6068, 0.2310, 0.8274, 0.1398, 0.2589, 0.0064,
        0.9455, 0.7121, 0.0986, 0.9549, 0.4726, 0.1366, 0.3290, 0.9323, 0.3228,
        0.9583, 0.7089, 0.1107, 0.9743, 0.1498, 0.2139, 0.3216, 0.2991, 0.7843,
        0.0489, 0.6093, 0.2060, 0.9702, 0.2499, 0.4500, 0.3086, 0.7304, 0.1349,
        0.6589, 0.9780, 0.2922, 0.5505, 0.2973, 0.6329, 0.6883, 0.9015, 0.5825,
        0.5057, 0.3467, 0.2415, 0.9832, 0.5247, 0.4903, 0.4895, 0.6200, 0.8065,
        0.0138, 0.3026, 0.2096, 0.1151, 0.5139, 0.1149, 0.2097, 0.4824, 0.9211,
        0.5255, 0.0536, 0.2820, 0.3395, 0.2058, 0.5210, 0.9748, 0.8561, 0.0685,
        0.7730, 0.0811, 0.4720, 0.7002, 0.3888, 0.8126, 0.5765, 0.5980, 0.3293,
        0.1374, 0.0565, 0.8223, 0.7646, 0.7631, 0.1598, 0.5843, 0.2414, 0.5108,
        0.2485, 0.5412, 0.6875, 0.9469, 0.4426, 0.9680, 0.0951, 0.0719, 0.1942,
        0.5659, 0.5737, 0.7803, 0.9762, 0.8326, 0.4935, 0.3197, 0.3643, 0.1375,
        0.5637, 0.5674, 0.9025, 0.3797, 0.7049, 0.0307, 0.9683, 0.1049, 0.7105,
        0.6966, 0.2000, 0.7896, 0.4336, 0.6025, 0.0066, 0.2166, 0.5604, 0.0782,
        0.6640, 0.9988, 0.3443, 0.0446, 0.2391, 0.2417, 0.2660, 0.6454, 0.8708,
        0.4188, 0.4120, 0.6376, 0.5463, 0.5214, 0.0786, 0.4010, 0.5928, 0.0247,
        0.2836, 0.1594, 0.4144, 0.6558, 0.6402, 0.4226, 0.0785, 0.1841, 0.4384,
        0.1776, 0.1451, 0.6120, 0.3095, 0.6375, 0.7446, 0.1182, 0.1442, 0.3297,
        0.3180, 0.4943, 0.0830, 0.9445, 0.2686, 0.1866, 0.1484, 0.1744, 0.7244,
        0.8358, 0.2060, 0.3809, 0.3305, 0.9919, 0.9469, 0.1970, 0.3224, 0.0704,
        0.4826, 0.6408, 0.7245, 0.6048, 0.5362, 0.8191, 0.5939, 0.8319, 0.5618,
        0.8720, 0.3567, 0.5021, 0.9448, 0.9247, 0.9510, 0.9533, 0.4388, 0.8670,
        0.2965, 0.3630, 0.8519, 0.5546, 0.3376, 0.7784, 0.5577, 0.9188, 0.6387,
        0.8628, 0.2464, 0.2945, 0.8249, 0.1407, 0.1416, 0.6298, 0.1568, 0.5900,
        0.0638, 0.4703, 0.9121, 0.9751, 0.8979, 0.6738, 0.8780, 0.0899, 0.9201,
        0.4437, 0.3468, 0.2314, 0.2427, 0.4305, 0.3386, 0.6811, 0.8609, 0.0356,
        0.7539, 0.8758, 0.3235, 0.0824, 0.3943, 0.7316, 0.3061, 0.3274, 0.3181,
        0.3651, 0.9916, 0.6031, 0.7193, 0.1152, 0.0696, 0.8872, 0.6175, 0.7566,
        0.4039, 0.0487, 0.9853, 0.5459, 0.7245, 0.0430, 0.0615, 0.3766, 0.7341,
        0.8894, 0.7512, 0.5459, 0.2261, 0.7005, 0.3395, 0.5303, 0.9922, 0.7489,
        0.9099, 0.1217, 0.5251, 0.3295, 0.4601, 0.7141, 0.4713, 0.8542, 0.1361,
        0.1830, 0.8641, 0.2778, 0.2382, 0.4201, 0.0562, 0.2713, 0.8479, 0.6643,
        0.9395, 0.2888, 0.6883, 0.4036, 0.1555, 0.0448, 0.4008, 0.9673, 0.0559,
        0.6019, 0.8909, 0.4629, 0.5184, 0.4910, 0.8696, 0.5682, 0.7485, 0.4645,
        0.0298, 0.0472, 0.2781, 0.1847, 0.1773, 0.0430, 0.5035, 0.4759, 0.1632,
        0.4727, 0.3241, 0.7979, 0.5776, 0.9485, 0.1808, 0.5953, 0.8802, 0.0991,
        0.1575, 0.5212, 0.2861, 0.1629, 0.8142, 0.3456, 0.4308, 0.4075, 0.5161,
        0.6445, 0.0716, 0.0679, 0.3143, 0.3814, 0.5265, 0.1770, 0.8199, 0.3430,
        0.6406, 0.0228, 0.7625, 0.8291, 0.7463, 0.5670, 0.1229, 0.8211, 0.5131,
        0.8990, 0.8715, 0.1674, 0.5084, 0.3622, 0.1882, 0.0809, 0.4150, 0.3598,
        0.5441, 0.3801, 0.5160, 0.4154, 0.3368, 0.8605, 0.3318, 0.3903, 0.5671,
        0.5602, 0.3410, 0.3019, 0.8924, 0.9533, 0.9099, 0.7521, 0.0489, 0.5392,
        0.1047, 0.1826, 0.8279, 0.0402, 0.8100, 0.8198, 0.2225, 0.3958, 0.9015,
        0.4661, 0.5805, 0.9626, 0.2403, 0.3884, 0.4673, 0.4590, 0.4446, 0.7121,
        0.7741, 0.2666, 0.3024, 0.0251, 0.7285, 0.6525, 0.0041, 0.0657, 0.3408,
        0.5322, 0.0776, 0.1887, 0.4864, 0.1112, 0.7864, 0.9571, 0.8367, 0.9944,
        0.9579, 0.8367, 0.5630, 0.6508, 0.5498, 0.8835, 0.3218, 0.4328, 0.5090,
        0.6201, 0.0993, 0.9176, 0.2918, 0.1857, 0.7410, 0.0047, 0.2669, 0.8563,
        0.6638, 0.7161, 0.9124, 0.0570, 0.9296, 0.7503, 0.5450, 0.1763, 0.2975,
        0.3667, 0.9487, 0.8542, 0.0637, 0.6769, 0.0878, 0.5272, 0.4628, 0.7728,
        0.2056, 0.1469, 0.5368, 0.8451, 0.3211, 0.6767, 0.6281, 0.4258, 0.9638,
        0.7799, 0.2563, 0.2662, 0.4799, 0.9032, 0.7659, 0.6708, 0.0968, 0.6357,
        0.6618, 0.2141, 0.1434, 0.1385, 0.3626, 0.5992, 0.7077, 0.2352, 0.9396,
        0.7730, 0.6471, 0.9458, 0.9779, 0.0400, 0.6695, 0.5889, 0.0042, 0.0446,
        0.3567, 0.0041, 0.4698, 0.0531, 0.7566, 0.3662, 0.6831, 0.8097, 0.8216,
        0.3287, 0.9199, 0.2125, 0.5333, 0.7237, 0.4768, 0.9945, 0.3674, 0.6263,
        0.3884, 0.0153, 0.1061, 0.0612, 0.7728, 0.8015, 0.5020, 0.1263, 0.8020,
        0.4555, 0.8181, 0.5180, 0.3248, 0.3046, 0.8839, 0.1501, 0.0677, 0.0645,
        0.8302, 0.3658, 0.8553, 0.5203, 0.6235, 0.2816, 0.7187, 0.0632, 0.3215,
        0.3471, 0.8103, 0.6826, 0.6804, 0.8577, 0.1603, 0.3107, 0.2977, 0.1387,
        0.2483, 0.4466, 0.6058, 0.8091, 0.9209, 0.4315, 0.9332, 0.4280, 0.2713,
        0.0815, 0.1326, 0.7730, 0.5825, 0.6632, 0.6605, 0.0857, 0.6526, 0.8940,
        0.3583, 0.1263, 0.9057, 0.8402, 0.1648, 0.1034, 0.8769, 0.1059, 0.8243,
        0.3036, 0.9903, 0.9349, 0.4232, 0.1171, 0.6476, 0.2477, 0.9493, 0.3656,
        0.5260, 0.3678, 0.4929, 0.1046, 0.7673, 0.5378, 0.1228, 0.1770, 0.9268,
        0.1053, 0.7606, 0.5018, 0.0645, 0.5230, 0.0068, 0.2736, 0.2494, 0.6741,
        0.5383, 0.3417, 0.7886, 0.0738, 0.1586, 0.2285, 0.2712, 0.5006, 0.3386,
        0.1294, 0.3547, 0.6119, 0.3975, 0.3873, 0.2488, 0.6903, 0.7849, 0.2772,
        0.3607, 0.7407, 0.9647, 0.9955, 0.2524, 0.1593, 0.8835, 0.4173, 0.3255,
        0.6960, 0.6278, 0.4636, 0.4471, 0.2504, 0.4842, 0.3870, 0.4917, 0.2941,
        0.1419, 0.7948, 0.9968, 0.6509, 0.4217, 0.3918, 0.9275, 0.9022, 0.1280,
        0.6053, 0.2950, 0.5459, 0.0483, 0.7723, 0.0608, 0.9211, 0.5496, 0.2009,
        0.8008, 0.5634, 0.4750, 0.6642, 0.4627, 0.2843, 0.2796, 0.9198, 0.5882,
        0.2988, 0.1688, 0.5271, 0.4091, 0.7875, 0.5803, 0.1633, 0.8787, 0.8174,
        0.4967, 0.7057, 0.0454, 0.3843, 0.0940, 0.6944, 0.7963, 0.3614, 0.8037,
        0.4419, 0.1307, 0.2887, 0.1208, 0.2176, 0.9910, 0.1800, 0.5128, 0.7257,
        0.8036, 0.3380, 0.9307, 0.6645, 0.1134, 0.4619, 0.9145, 0.2078, 0.8331,
        0.9213, 0.5730, 0.8720, 0.5022, 0.7501, 0.7396, 0.8787, 0.3669, 0.3008,
        0.4407, 0.5886, 0.9671, 0.8588, 0.9337, 0.1005, 0.7053, 0.2850, 0.1750,
        0.3186])

weight_1 = torch.rand(784, 200)
weight_2 = torch.rand(200, 10)

# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)

# Multiply hidden_1 with weight_2
output_layer = torch.matmul(hidden_1 , weight_2)
print(output_layer)

net = Net()
output2 = net.forward(input_layer)
print(output2)
