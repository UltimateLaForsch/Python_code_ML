# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:39:52 2021

@author: Ultimate LaForsch
"""

import numpy as np
import tensorflow as tf
from tensorflow import reshape

gray_tensor = np.array([[187, 189, 190, 192, 195, 198, 198, 198, 200, 200, 201, 205, 206,
        203, 206, 206, 206, 207, 209, 207, 205, 205, 208, 208, 206, 207,
        206, 206],
       [189, 191, 193, 195, 197, 199, 200, 199, 200, 201, 202, 206, 206,
        208, 204, 204, 210, 209, 208, 208, 207, 206, 208, 208, 207, 207,
        207, 207],
       [189, 192, 195, 198, 198, 201, 202, 203, 205, 206, 205, 209, 207,
        204, 211, 210, 205, 208, 211, 208, 206, 207, 209, 210, 210, 208,
        210, 210],
       [191, 192, 195, 197, 199, 199, 204, 201, 203, 208, 206, 207, 209,
        207, 213, 208, 209, 211, 221, 205, 204, 239, 182, 212, 213, 212,
        209, 209],
       [193, 195, 195, 199, 201, 201, 203, 205, 206, 216, 223, 203, 207,
        239, 225, 193, 188, 230, 232, 195, 176, 239, 191, 205, 215, 212,
        211, 213],
       [194, 196, 199, 200, 206, 202, 205, 207, 196, 255, 238, 197, 160,
        235, 226, 191, 146, 190, 226, 201, 160, 228, 211, 162, 215, 214,
        216, 213],
       [195, 198, 201, 202, 202, 203, 232, 215, 197, 246, 230, 215, 153,
        235, 221, 205, 162, 181, 224, 210, 166, 202, 209, 163, 190, 213,
        211, 213],
       [196, 198, 201, 208, 202, 243, 240, 215, 166, 246, 233, 223, 171,
        235, 212, 214, 165, 206, 224, 199, 152, 126, 206, 199, 170, 165,
        214, 215],
       [198, 201, 197, 204, 189, 247, 244, 230, 206, 246, 235, 220, 188,
        235, 211, 218, 142, 227, 223, 176, 152,  74, 192, 206, 200, 145,
        177, 215],
       [200, 203, 206, 206, 236, 254, 254, 233, 215, 205, 242, 219, 185,
        234, 230, 223, 131, 201, 209, 156, 141,  83, 175, 220, 196, 157,
        185, 222],
       [201, 200, 201, 193, 253, 231, 245, 246, 209, 159, 241, 214, 176,
        219, 234, 212, 133, 132, 175, 149, 109, 100, 225, 226, 209, 147,
        219, 221],
       [202, 203, 203, 196, 253, 209, 241, 233, 194, 150, 234, 204, 174,
        160, 208, 189, 146, 101, 172, 145,  76, 195, 230, 226, 194, 121,
        227, 224],
       [204, 203, 210, 245, 251, 222, 207, 198, 152, 112, 207, 171, 163,
         97, 163, 154, 122, 105, 175, 169, 175, 227, 226, 206, 154, 147,
        219, 225],
       [204, 205, 201, 250, 246, 217, 167, 204, 146, 116, 192, 170, 161,
         78, 151, 165, 115, 181, 228, 225, 223, 215, 203, 181, 144, 202,
        220, 227],
       [205, 207, 198, 252, 254, 228, 198, 185, 162, 128, 202, 194, 144,
         64, 135, 155, 237, 241, 237, 226, 211, 177, 179, 155, 142, 227,
        227, 228],
       [210, 211, 208, 255, 252, 240, 219, 187, 168, 148, 187, 202, 151,
        103, 192, 246, 253, 244, 233, 221, 199, 179, 157, 154, 116, 234,
        228, 231],
       [208, 211, 209, 254, 254, 248, 231, 216, 193, 175, 178, 201, 208,
        240, 253, 254, 249, 238, 222, 206, 185, 160, 143, 143, 214, 231,
        230, 230],
       [209, 212, 205, 254, 254, 252, 241, 229, 217, 187, 207, 224, 249,
        253, 251, 250, 242, 228, 206, 183, 166, 150, 143, 172, 229, 234,
        235, 230],
       [208, 211, 206, 254, 254, 255, 249, 238, 231, 211, 213, 230, 250,
        254, 252, 246, 233, 217, 188, 164, 150, 143, 120, 235, 231, 230,
        231, 231],
       [209, 213, 211, 253, 255, 255, 252, 244, 233, 222, 217, 224, 246,
        251, 242, 234, 225, 195, 173, 153, 134, 116, 225, 232, 235, 232,
        233, 233],
       [209, 214, 214, 246, 254, 253, 252, 240, 224, 214, 213, 217, 233,
        233, 230, 214, 199, 190, 150, 145, 127, 201, 233, 234, 232, 234,
        233, 234],
       [211, 215, 215, 243, 254, 254, 245, 232, 221, 208, 213, 218, 225,
        223, 206, 195, 169, 157, 132, 126, 170, 238, 234, 235, 234, 234,
        234, 234],
       [214, 216, 217, 209, 254, 250, 236, 229, 212, 197, 206, 210, 221,
        210, 196, 170, 148, 140, 118, 134, 240, 235, 234, 235, 235, 236,
        235, 236],
       [186, 175, 180, 150, 156, 158, 144, 124, 132, 134, 148, 153, 150,
        146, 137, 134, 126, 109, 114, 235, 237, 234, 238, 236, 236, 236,
        236, 237],
       [145, 135, 137, 134, 122, 136, 112,  95,  94,  90,  93,  65,  60,
         66,  61,  66,  58,  66,  80, 164, 247, 235, 236, 237, 239, 237,
        237, 235],
       [140, 146, 136, 132, 129, 134, 100, 103, 100, 100,  87,  64,  66,
         65,  57,  57,  61,  61,  64,  65, 177, 242, 238, 238, 239, 238,
        238, 238],
       [141, 146, 140, 131, 130, 136,  93,  97, 102,  96,  78,  71,  68,
         64,  60,  61,  60,  55,  58,  48, 254, 238, 240, 239, 238, 237,
        237, 238],
       [146, 143, 137, 138, 129, 113,  94,  98, 101,  87,  75,  70,  68,
         63,  60,  58,  56,  57,  63,  81, 237, 237, 240, 240, 239, 240,
        240, 240]])
color_tensor = np.array([[[190., 185., 191.],
        [190., 187., 196.],
        [191., 188., 195.],
        [192., 191., 196.],
        [195., 194., 199.],
        [198., 197., 202.],
        [199., 197., 202.],
        [201., 196., 202.],
        [203., 197., 207.],
        [203., 197., 207.],
        [204., 198., 208.],
        [206., 203., 210.],
        [206., 205., 210.],
        [203., 202., 207.],
        [206., 206., 208.],
        [209., 204., 210.],
        [206., 205., 213.],
        [208., 205., 214.],
        [209., 208., 214.],
        [207., 206., 212.],
        [206., 204., 209.],
        [211., 202., 205.],
        [211., 207., 206.],
        [209., 206., 213.],
        [210., 202., 213.],
        [208., 205., 212.],
        [206., 205., 210.],
        [206., 205., 210.]],

       [[192., 187., 193.],
        [192., 189., 198.],
        [194., 191., 198.],
        [196., 194., 199.],
        [197., 196., 201.],
        [200., 198., 203.],
        [201., 198., 205.],
        [200., 197., 204.],
        [203., 198., 205.],
        [204., 198., 208.],
        [203., 200., 211.],
        [207., 204., 215.],
        [206., 205., 213.],
        [208., 207., 213.],
        [204., 203., 208.],
        [205., 202., 209.],
        [209., 209., 217.],
        [208., 208., 216.],
        [208., 207., 213.],
        [208., 207., 212.],
        [208., 206., 209.],
        [209., 205., 206.],
        [211., 207., 208.],
        [208., 207., 215.],
        [207., 206., 214.],
        [207., 206., 212.],
        [207., 206., 211.],
        [207., 206., 211.]],

       [[190., 188., 193.],
        [193., 190., 199.],
        [196., 194., 199.],
        [199., 197., 202.],
        [199., 197., 200.],
        [202., 200., 205.],
        [203., 201., 206.],
        [203., 202., 208.],
        [206., 204., 209.],
        [207., 204., 215.],
        [208., 202., 214.],
        [212., 206., 216.],
        [208., 205., 212.],
        [205., 203., 208.],
        [215., 208., 215.],
        [214., 208., 212.],
        [205., 204., 209.],
        [207., 208., 213.],
        [212., 210., 213.],
        [212., 206., 208.],
        [212., 203., 204.],
        [211., 206., 203.],
        [215., 207., 205.],
        [213., 208., 214.],
        [207., 210., 219.],
        [207., 208., 213.],
        [207., 211., 214.],
        [210., 209., 214.]],

       [[192., 190., 195.],
        [193., 190., 199.],
        [198., 193., 200.],
        [200., 195., 201.],
        [202., 197., 201.],
        [203., 198., 202.],
        [205., 203., 208.],
        [202., 199., 206.],
        [204., 202., 207.],
        [211., 206., 213.],
        [212., 202., 211.],
        [213., 203., 211.],
        [215., 206., 209.],
        [213., 204., 205.],
        [221., 209., 211.],
        [218., 204., 203.],
        [213., 208., 205.],
        [215., 210., 206.],
        [228., 218., 216.],
        [216., 201., 198.],
        [217., 200., 193.],
        [247., 237., 227.],
        [195., 177., 173.],
        [220., 208., 210.],
        [212., 213., 217.],
        [211., 212., 217.],
        [208., 209., 213.],
        [208., 209., 213.]],

       [[193., 192., 197.],
        [196., 193., 202.],
        [198., 192., 202.],
        [202., 197., 203.],
        [204., 199., 203.],
        [204., 200., 201.],
        [206., 201., 205.],
        [208., 203., 209.],
        [206., 205., 211.],
        [218., 213., 217.],
        [230., 220., 221.],
        [210., 200., 199.],
        [215., 204., 202.],
        [247., 236., 232.],
        [236., 221., 218.],
        [208., 188., 181.],
        [198., 185., 177.],
        [240., 227., 221.],
        [243., 229., 220.],
        [214., 187., 180.],
        [195., 168., 157.],
        [251., 235., 219.],
        [211., 182., 174.],
        [220., 199., 196.],
        [215., 215., 215.],
        [211., 212., 217.],
        [211., 210., 215.],
        [214., 212., 217.]],

       [[195., 193., 198.],
        [196., 195., 203.],
        [200., 197., 206.],
        [201., 199., 204.],
        [212., 203., 208.],
        [206., 200., 202.],
        [206., 205., 201.],
        [211., 205., 205.],
        [199., 195., 196.],
        [255., 255., 253.],
        [252., 232., 234.],
        [212., 191., 188.],
        [181., 150., 148.],
        [249., 230., 226.],
        [244., 218., 217.],
        [211., 183., 172.],
        [165., 138., 129.],
        [205., 185., 178.],
        [240., 220., 211.],
        [228., 190., 177.],
        [186., 148., 135.],
        [247., 221., 208.],
        [235., 200., 196.],
        [181., 153., 149.],
        [221., 214., 206.],
        [214., 213., 218.],
        [215., 216., 220.],
        [213., 212., 217.]],

       [[196., 194., 199.],
        [198., 197., 205.],
        [201., 200., 208.],
        [203., 201., 206.],
        [205., 200., 204.],
        [206., 202., 201.],
        [238., 231., 225.],
        [226., 211., 206.],
        [205., 196., 189.],
        [249., 246., 239.],
        [241., 226., 223.],
        [232., 208., 204.],
        [177., 142., 138.],
        [249., 230., 223.],
        [236., 215., 210.],
        [231., 194., 188.],
        [185., 152., 143.],
        [196., 176., 169.],
        [240., 217., 209.],
        [239., 197., 183.],
        [195., 153., 139.],
        [226., 192., 180.],
        [237., 196., 192.],
        [187., 152., 148.],
        [204., 185., 178.],
        [219., 210., 211.],
        [214., 210., 211.],
        [213., 213., 215.]],

       [[197., 195., 200.],
        [198., 197., 203.],
        [201., 200., 206.],
        [209., 207., 212.],
        [202., 202., 202.],
        [246., 242., 239.],
        [247., 237., 235.],
        [233., 208., 204.],
        [182., 160., 147.],
        [250., 247., 232.],
        [247., 227., 226.],
        [239., 216., 208.],
        [195., 160., 156.],
        [246., 232., 223.],
        [229., 205., 201.],
        [238., 203., 197.],
        [184., 158., 145.],
        [219., 201., 191.],
        [241., 219., 208.],
        [229., 186., 169.],
        [184., 137., 121.],
        [158., 110.,  98.],
        [238., 191., 181.],
        [229., 186., 180.],
        [194., 159., 155.],
        [180., 159., 156.],
        [221., 211., 210.],
        [215., 215., 215.]],

       [[199., 197., 202.],
        [201., 200., 206.],
        [197., 196., 202.],
        [205., 203., 208.],
        [190., 189., 187.],
        [248., 247., 243.],
        [250., 242., 239.],
        [248., 222., 221.],
        [225., 198., 189.],
        [253., 245., 232.],
        [249., 230., 226.],
        [236., 213., 205.],
        [211., 178., 171.],
        [244., 232., 220.],
        [227., 204., 198.],
        [241., 208., 199.],
        [162., 134., 120.],
        [240., 223., 207.],
        [240., 217., 203.],
        [209., 163., 147.],
        [187., 135., 121.],
        [107.,  55.,  44.],
        [224., 177., 167.],
        [234., 194., 186.],
        [226., 187., 180.],
        [162., 137., 132.],
        [187., 173., 173.],
        [216., 214., 217.]],

       [[201., 199., 204.],
        [203., 202., 208.],
        [206., 205., 210.],
        [209., 204., 210.],
        [237., 236., 234.],
        [255., 253., 254.],
        [254., 254., 252.],
        [244., 228., 229.],
        [234., 206., 202.],
        [218., 201., 194.],
        [255., 238., 231.],
        [233., 213., 202.],
        [208., 175., 166.],
        [243., 231., 217.],
        [244., 224., 215.],
        [246., 214., 203.],
        [160., 118., 104.],
        [227., 190., 174.],
        [236., 198., 185.],
        [187., 142., 123.],
        [176., 125., 108.],
        [116.,  64.,  51.],
        [201., 164., 148.],
        [241., 212., 198.],
        [224., 183., 177.],
        [177., 148., 142.],
        [193., 181., 183.],
        [222., 221., 226.]],

       [[202., 200., 205.],
        [201., 199., 204.],
        [202., 200., 205.],
        [194., 192., 195.],
        [254., 252., 253.],
        [237., 228., 229.],
        [246., 245., 240.],
        [253., 244., 237.],
        [231., 200., 195.],
        [175., 153., 142.],
        [252., 238., 229.],
        [234., 205., 197.],
        [203., 165., 154.],
        [235., 213., 200.],
        [248., 228., 221.],
        [237., 200., 191.],
        [168., 116., 103.],
        [163., 117., 101.],
        [208., 161., 143.],
        [194., 127., 111.],
        [146.,  88.,  74.],
        [125.,  88.,  72.],
        [246., 217., 203.],
        [245., 218., 207.],
        [233., 198., 192.],
        [167., 138., 132.],
        [226., 216., 215.],
        [222., 220., 225.]],

       [[203., 200., 207.],
        [204., 202., 207.],
        [204., 202., 205.],
        [197., 195., 198.],
        [254., 253., 249.],
        [222., 204., 202.],
        [251., 238., 229.],
        [247., 227., 216.],
        [225., 180., 174.],
        [177., 137., 129.],
        [253., 227., 214.],
        [230., 194., 180.],
        [203., 160., 141.],
        [188., 146., 130.],
        [232., 198., 186.],
        [221., 175., 159.],
        [188., 126., 113.],
        [137.,  81.,  66.],
        [208., 155., 141.],
        [186., 123., 114.],
        [108.,  56.,  43.],
        [218., 184., 174.],
        [247., 223., 213.],
        [246., 217., 211.],
        [218., 183., 177.],
        [138., 113., 109.],
        [233., 225., 223.],
        [225., 223., 228.]],

       [[204., 203., 208.],
        [203., 202., 207.],
        [211., 209., 214.],
        [245., 245., 247.],
        [254., 251., 246.],
        [238., 214., 210.],
        [226., 199., 188.],
        [222., 189., 170.],
        [191., 133., 122.],
        [148.,  91.,  80.],
        [243., 191., 178.],
        [205., 156., 141.],
        [201., 144., 124.],
        [136.,  75.,  57.],
        [205., 143., 132.],
        [193., 134., 118.],
        [165.,  98.,  89.],
        [142.,  85.,  76.],
        [208., 160., 156.],
        [201., 153., 151.],
        [202., 162., 154.],
        [247., 218., 212.],
        [245., 217., 213.],
        [229., 196., 191.],
        [177., 144., 137.],
        [162., 141., 140.],
        [222., 218., 217.],
        [225., 224., 230.]],

       [[204., 203., 208.],
        [205., 204., 209.],
        [202., 200., 205.],
        [250., 250., 252.],
        [249., 245., 242.],
        [234., 209., 204.],
        [192., 156., 144.],
        [236., 190., 174.],
        [190., 123., 114.],
        [157.,  93.,  81.],
        [235., 173., 160.],
        [209., 151., 137.],
        [203., 141., 120.],
        [118.,  51.,  34.],
        [196., 128., 117.],
        [207., 143., 131.],
        [150.,  97.,  91.],
        [207., 168., 163.],
        [246., 221., 217.],
        [245., 217., 214.],
        [243., 214., 210.],
        [236., 205., 202.],
        [223., 194., 188.],
        [202., 172., 164.],
        [164., 135., 129.],
        [215., 197., 197.],
        [223., 218., 222.],
        [227., 226., 234.]],

       [[205., 204., 209.],
        [207., 206., 211.],
        [198., 197., 202.],
        [252., 252., 254.],
        [255., 254., 250.],
        [246., 220., 219.],
        [226., 186., 178.],
        [219., 170., 156.],
        [203., 141., 130.],
        [165., 109.,  96.],
        [238., 186., 173.],
        [230., 178., 167.],
        [182., 124., 110.],
        [ 97.,  39.,  25.],
        [170., 116., 104.],
        [183., 142., 136.],
        [251., 232., 225.],
        [251., 237., 234.],
        [248., 232., 233.],
        [245., 216., 218.],
        [232., 202., 202.],
        [197., 168., 162.],
        [200., 170., 162.],
        [175., 147., 136.],
        [161., 134., 127.],
        [237., 223., 223.],
        [230., 225., 232.],
        [228., 227., 235.]],

       [[210., 209., 214.],
        [211., 211., 213.],
        [208., 208., 210.],
        [255., 255., 255.],
        [252., 252., 250.],
        [250., 236., 235.],
        [242., 209., 200.],
        [220., 172., 158.],
        [207., 149., 135.],
        [183., 130., 116.],
        [223., 171., 158.],
        [239., 185., 173.],
        [190., 131., 123.],
        [130.,  87.,  78.],
        [211., 184., 175.],
        [254., 243., 239.],
        [253., 254., 248.],
        [248., 243., 239.],
        [247., 227., 229.],
        [239., 213., 214.],
        [217., 192., 188.],
        [199., 170., 166.],
        [177., 148., 142.],
        [171., 146., 139.],
        [128., 111., 103.],
        [238., 232., 234.],
        [231., 226., 233.],
        [231., 230., 238.]],

       [[209., 207., 210.],
        [212., 210., 213.],
        [209., 209., 209.],
        [254., 254., 254.],
        [254., 254., 254.],
        [252., 247., 244.],
        [245., 225., 218.],
        [239., 205., 193.],
        [225., 179., 164.],
        [208., 161., 145.],
        [208., 165., 146.],
        [226., 191., 172.],
        [231., 199., 188.],
        [251., 236., 229.],
        [255., 253., 246.],
        [255., 253., 254.],
        [255., 246., 245.],
        [248., 234., 233.],
        [239., 215., 211.],
        [224., 196., 192.],
        [205., 178., 171.],
        [179., 151., 148.],
        [160., 135., 131.],
        [155., 138., 131.],
        [222., 211., 205.],
        [232., 230., 233.],
        [231., 228., 237.],
        [230., 229., 237.]],

       [[212., 207., 211.],
        [213., 211., 214.],
        [205., 205., 205.],
        [254., 254., 254.],
        [254., 254., 254.],
        [253., 252., 250.],
        [251., 238., 232.],
        [246., 222., 212.],
        [241., 207., 195.],
        [216., 173., 157.],
        [234., 196., 177.],
        [240., 219., 202.],
        [255., 247., 236.],
        [255., 252., 251.],
        [251., 251., 253.],
        [250., 250., 248.],
        [250., 238., 238.],
        [243., 222., 221.],
        [223., 198., 193.],
        [200., 176., 166.],
        [185., 158., 151.],
        [167., 142., 138.],
        [157., 138., 134.],
        [180., 169., 165.],
        [233., 228., 225.],
        [234., 234., 236.],
        [236., 233., 240.],
        [230., 229., 237.]],

       [[209., 207., 210.],
        [212., 210., 213.],
        [206., 206., 206.],
        [254., 254., 254.],
        [254., 254., 254.],
        [254., 255., 255.],
        [252., 248., 245.],
        [249., 234., 227.],
        [246., 224., 211.],
        [235., 202., 183.],
        [239., 202., 186.],
        [244., 226., 216.],
        [254., 249., 245.],
        [253., 254., 255.],
        [253., 251., 254.],
        [250., 245., 242.],
        [247., 227., 226.],
        [235., 210., 206.],
        [205., 181., 171.],
        [183., 156., 147.],
        [167., 142., 135.],
        [157., 138., 132.],
        [130., 116., 113.],
        [238., 234., 233.],
        [231., 231., 231.],
        [231., 229., 232.],
        [231., 230., 236.],
        [231., 230., 238.]],

       [[212., 207., 211.],
        [214., 212., 215.],
        [211., 211., 211.],
        [253., 253., 253.],
        [255., 255., 253.],
        [255., 255., 253.],
        [252., 253., 248.],
        [250., 243., 233.],
        [247., 228., 214.],
        [242., 214., 200.],
        [240., 208., 195.],
        [243., 215., 212.],
        [252., 245., 239.],
        [252., 251., 246.],
        [249., 240., 235.],
        [248., 228., 227.],
        [239., 220., 214.],
        [211., 188., 182.],
        [192., 165., 158.],
        [169., 146., 138.],
        [148., 128., 119.],
        [128., 113., 110.],
        [232., 222., 220.],
        [233., 231., 232.],
        [235., 234., 239.],
        [235., 231., 232.],
        [236., 231., 238.],
        [236., 230., 240.]],

       [[210., 208., 211.],
        [214., 214., 216.],
        [213., 214., 216.],
        [246., 245., 250.],
        [254., 254., 254.],
        [253., 253., 251.],
        [255., 252., 245.],
        [250., 236., 225.],
        [243., 217., 204.],
        [238., 204., 192.],
        [237., 203., 193.],
        [237., 208., 200.],
        [246., 228., 218.],
        [244., 229., 222.],
        [245., 225., 218.],
        [233., 206., 199.],
        [217., 189., 185.],
        [210., 180., 178.],
        [170., 141., 137.],
        [162., 137., 130.],
        [141., 121., 114.],
        [209., 198., 194.],
        [236., 232., 233.],
        [234., 233., 241.],
        [232., 230., 241.],
        [235., 232., 239.],
        [234., 231., 242.],
        [235., 232., 243.]],

       [[212., 210., 213.],
        [215., 215., 217.],
        [215., 214., 219.],
        [243., 242., 247.],
        [255., 253., 254.],
        [255., 254., 249.],
        [251., 244., 236.],
        [245., 227., 215.],
        [241., 213., 202.],
        [232., 198., 189.],
        [239., 202., 193.],
        [242., 207., 201.],
        [246., 216., 208.],
        [242., 214., 210.],
        [225., 198., 191.],
        [216., 186., 178.],
        [190., 159., 154.],
        [177., 148., 144.],
        [151., 123., 119.],
        [139., 120., 114.],
        [180., 167., 161.],
        [241., 237., 238.],
        [235., 233., 238.],
        [235., 234., 242.],
        [235., 232., 243.],
        [234., 232., 243.],
        [234., 233., 241.],
        [235., 232., 239.]],

       [[215., 213., 214.],
        [217., 215., 218.],
        [217., 217., 217.],
        [210., 208., 209.],
        [255., 253., 252.],
        [254., 249., 245.],
        [244., 234., 225.],
        [242., 224., 212.],
        [231., 204., 193.],
        [218., 187., 182.],
        [229., 196., 189.],
        [231., 200., 195.],
        [241., 212., 208.],
        [231., 200., 197.],
        [217., 186., 183.],
        [190., 161., 153.],
        [168., 139., 131.],
        [159., 132., 123.],
        [134., 111., 103.],
        [145., 130., 125.],
        [244., 238., 240.],
        [236., 234., 239.],
        [234., 233., 241.],
        [235., 234., 242.],
        [236., 233., 242.],
        [236., 235., 243.],
        [235., 233., 244.],
        [237., 234., 243.]],

       [[189., 185., 182.],
        [176., 175., 170.],
        [183., 180., 173.],
        [154., 150., 141.],
        [162., 155., 147.],
        [164., 157., 147.],
        [150., 142., 129.],
        [133., 121., 107.],
        [144., 127., 117.],
        [145., 128., 120.],
        [158., 144., 135.],
        [163., 148., 141.],
        [163., 146., 139.],
        [159., 140., 134.],
        [150., 131., 125.],
        [147., 129., 119.],
        [139., 121., 107.],
        [121., 105.,  90.],
        [123., 111.,  99.],
        [241., 233., 230.],
        [237., 237., 239.],
        [234., 233., 239.],
        [238., 237., 243.],
        [236., 235., 243.],
        [237., 234., 241.],
        [236., 235., 241.],
        [236., 235., 243.],
        [238., 235., 244.]],

       [[149., 145., 136.],
        [138., 136., 124.],
        [140., 137., 122.],
        [139., 133., 117.],
        [127., 121., 109.],
        [141., 135., 119.],
        [115., 112.,  95.],
        [ 98.,  95.,  80.],
        [ 99.,  92.,  82.],
        [ 96.,  89.,  81.],
        [ 99.,  92.,  84.],
        [ 71.,  64.,  56.],
        [ 67.,  58.,  51.],
        [ 73.,  64.,  57.],
        [ 68.,  59.,  52.],
        [ 73.,  63.,  54.],
        [ 65.,  55.,  43.],
        [ 73.,  64.,  49.],
        [ 85.,  78.,  68.],
        [167., 163., 160.],
        [248., 247., 245.],
        [236., 234., 239.],
        [237., 234., 241.],
        [238., 235., 242.],
        [239., 238., 243.],
        [237., 236., 242.],
        [237., 236., 242.],
        [236., 233., 240.]],

       [[144., 140., 128.],
        [147., 145., 130.],
        [139., 136., 121.],
        [135., 132., 115.],
        [134., 129., 110.],
        [137., 134., 117.],
        [103., 100.,  85.],
        [105., 103.,  88.],
        [103., 101.,  89.],
        [104., 100.,  91.],
        [ 93.,  86.,  78.],
        [ 68.,  64.,  55.],
        [ 72.,  65.,  57.],
        [ 71.,  64.,  56.],
        [ 64.,  55.,  48.],
        [ 63.,  56.,  48.],
        [ 67.,  59.,  48.],
        [ 67.,  59.,  48.],
        [ 69.,  62.,  54.],
        [ 71.,  63.,  61.],
        [180., 176., 175.],
        [245., 240., 246.],
        [239., 236., 243.],
        [238., 237., 243.],
        [239., 238., 244.],
        [238., 237., 243.],
        [238., 237., 245.],
        [238., 237., 243.]],

       [[144., 142., 130.],
        [148., 145., 130.],
        [145., 141., 132.],
        [133., 130., 115.],
        [133., 130., 111.],
        [139., 136., 119.],
        [ 96.,  93.,  78.],
        [100.,  96.,  85.],
        [108., 101.,  93.],
        [101.,  94.,  84.],
        [ 82.,  78.,  67.],
        [ 74.,  72.,  60.],
        [ 70.,  68.,  56.],
        [ 67.,  63.,  52.],
        [ 65.,  58.,  48.],
        [ 66.,  59.,  49.],
        [ 65.,  59.,  47.],
        [ 61.,  52.,  43.],
        [ 65.,  56.,  49.],
        [ 54.,  46.,  44.],
        [255., 253., 252.],
        [239., 237., 242.],
        [240., 239., 245.],
        [239., 238., 244.],
        [238., 237., 245.],
        [237., 236., 244.],
        [237., 235., 246.],
        [238., 237., 245.]],

       [[149., 146., 137.],
        [146., 142., 130.],
        [141., 137., 126.],
        [141., 138., 121.],
        [132., 129., 110.],
        [116., 113.,  96.],
        [ 97.,  93.,  81.],
        [104.,  97.,  87.],
        [107., 100.,  90.],
        [ 93.,  86.,  78.],
        [ 81.,  74.,  64.],
        [ 74.,  70.,  61.],
        [ 70.,  68.,  56.],
        [ 66.,  62.,  51.],
        [ 65.,  58.,  48.],
        [ 64.,  57.,  49.],
        [ 61.,  54.,  44.],
        [ 63.,  56.,  46.],
        [ 69.,  62.,  54.],
        [ 85.,  80.,  77.],
        [240., 236., 237.],
        [238., 236., 241.],
        [240., 239., 247.],
        [239., 239., 247.],
        [239., 238., 246.],
        [240., 239., 247.],
        [240., 239., 247.],
        [240., 239., 247.]]])

# Reshape the grayscale image tensor into a vector
gray_vector = reshape(gray_tensor, (-1, 1))

# Reshape the color image tensor into a vector
color_vector = reshape(color_tensor, (-1, 1))

