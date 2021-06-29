# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import subprocess
import os

from archai.common.utils import exec_shell_command

def main():
    parser = argparse.ArgumentParser(description='Proxynas Wrapper Nb101 Main')
    parser.add_argument('--algos', type=str, default='''
                                                        proxynas_nasbench101_space,                                                        
                                                        nb101_regular_eval,
                                                        freezeaddon_nasbench101_space,
                                                        ''',
                        help='NAS algos to run, separated by comma')
    parser.add_argument('--train-top1-acc-threshold', type=float)
    parser.add_argument('--arch-list-index', type=int)
    parser.add_argument('--num-archs', type=int)
    parser.add_argument('--datasets', type=str, default='cifar10')
    args, extra_args = parser.parse_known_args()

    # hard coded list of architectures to process
    all_archs = [308250, 372713, 127402, 327887, 192346, 56394, 166562, 151554, 211785, 380051, 243508, 288440, 195677, 269711,\
                321740, 206763, 167112, 275520, 224546, 380498, 131889, 175001, 25655, 60931, 163207, 168814, 61590, 111241,\
                181984, 349247, 127307, 408089, 370757, 226534, 216101, 18165, 21928, 63665, 238841, 124559, 378931, 407797,\
                136891, 410872, 305684, 236002, 283084, 245209, 66451, 12580, 191909, 152022, 258113, 332787, 79637, 398027,\
                283133, 277836, 205356, 184584, 405129, 405589, 249661, 16166, 316679, 6370, 391920, 421969, 274026, 379628,\
                126542, 120180, 378814, 58681, 38067, 246316, 304934, 83969, 17527, 215497, 135524, 302912, 322132, 422653, 363708,\
                17371, 50288, 330230, 358540, 235897, 189644, 406130, 230121, 27330, 200119, 23354, 355684, 108390, 110712, 62355,\
                216550, 143628, 332049, 362315, 405482, 418008, 200288, 201437, 394881, 362641, 79044, 213558, 123268, 216682, 85618,\
                321722, 23188, 210806, 214360, 185654, 277592, 89605, 368857, 177620, 205251, 224710, 322992, 420701, 378407, 69624,\
                137970, 163891, 218879, 41614, 128206, 321616, 159572, 147519, 227560, 268922, 244126, 186176, 25401, 270657, 348371,\
                166753, 95608, 55676, 242400, 374913, 274183, 75721, 33393, 161549, 307883, 420440, 396705, 157256, 136606, 395089,\
                3229, 49047, 370831, 144386, 61201, 136779, 204438, 420048, 419437, 148775, 198801, 191775, 266696, 9137, 178354, 188026,\
                402080, 331564, 140819, 178853, 303691, 56564, 238017, 69908, 256202, 267241, 375085, 245718, 75494, 240900, 403861,\
                389099, 322357, 394771, 18164, 218494, 250425, 200352, 104147, 186800, 204314, 98250, 156892, 306783, 227236, 93905,\
                127311, 231297, 162022, 219547, 330970, 95298, 348904, 264001, 17613, 232911, 163923, 307893, 108213, 250004, 46909,\
                346023, 105515, 130368, 334172, 343252, 238172, 45891, 285709, 315220, 155637, 380753, 280171, 155230, 89503, 18228,\
                169763, 323035, 180302, 251458, 300310, 267897, 287632, 32797, 254514, 57710, 378877, 317850, 99547, 159134, 244535,\
                24958, 130334, 52629, 246343, 18518, 21221, 323073, 392905, 336447, 410681, 207330, 379276, 264854, 202335, 104984, 103972,\
                57216, 225028, 244218, 187624, 58827, 268540, 146438, 161896, 157056, 128287, 43993, 370385, 306811, 146811, 252895, 203513,\
                64217, 207534, 406987, 380487, 199999, 9747, 50943, 168656, 378271, 11652, 160781, 413562, 71569, 38081, 209364, 10294, 61663,\
                70405, 129838, 155300, 345723, 342047, 170593, 396010, 213806, 145979, 189034, 146918, 162407, 330809, 320932, 373133, 334852,\
                76750, 280172, 401920, 109194, 132087, 265803, 107082, 77331, 235534, 113393, 235757, 177124, 97387, 25533, 413990, 247793,\
                129938, 355608, 220315, 240356, 156465, 199895, 389308, 209862, 171681, 410791, 244557, 188427, 80773, 349557, 332973, 30700,\
                203181, 377335, 290114, 317131, 148620, 313465, 92207, 212795, 390709, 107736, 38511, 256238, 217406, 314341, 100278, 104724,\
                140905, 45213, 49147, 84829, 245105, 129132, 79411, 416963, 76182, 421479, 50566, 67868, 328590, 197531, 84910, 294779, 10440,\
                264059, 171717, 278558, 360628, 153160, 53294, 406338, 155673, 24735, 34851, 121434, 143089, 227639, 196910, 16835, 136358,\
                132071, 141931, 379252, 177309, 279616, 324550, 210799, 330280, 172245, 47063, 361389, 256304, 63994, 281005, 83486, 212000,\
                202937, 117839, 409037, 207230, 124962, 107703, 407942, 173995, 289602, 28062, 333595, 227824, 136654, 13419, 361756, 269665,\
                300335, 271044, 24417, 378679, 293756, 271487, 195634, 258836, 51586, 392746, 138140, 26588, 333957, 410909, 205530, 109386, \
                13285, 96690, 205933, 197632, 270152, 60509, 52637, 297185, 135440, 222150, 399354, 362653, 325239, 237712, 46177, 94, 361030,\
                253594, 42410, 332303, 261105, 303581, 36056, 344612, 95847, 256822, 313505, 391540, 131288, 115995, 235411, 275398, 371221,\
                37828, 103940, 403995, 285218, 395603, 118096, 300290, 34092, 212652, 97737, 309936, 49492, 243827, 47206, 420983, 66468, 246662,\
                398055, 298113, 159752, 228033, 42913, 59447, 2513, 71722, 64244, 192136, 154933, 23167, 207274, 373130, 172911, 82760, 44935,\
                89508, 373225, 420352, 239371, 126026, 368066, 329842, 146463, 12804, 290930, 154934, 193887, 18061, 317949, 239223, 21345,\
                143834, 154976, 284901, 369797, 61704, 189846, 89804, 104646, 58470, 865, 407669, 111616, 88728, 127834, 242177, 290873, 83210,\
                274483, 16829, 226004, 56673, 101938, 33130, 73116, 390339, 666, 306143, 420484, 87254, 405120, 238075, 311922, 15418, 419022, 58623,\
                238528, 230766, 380988, 352991, 187332, 16871, 9899, 192980, 309186, 21436, 5472, 231720, 232660, 304681, 280173, 377163, 378070,\
                105458, 298599, 333960, 180888, 366658, 54179, 308719, 327221, 153363, 343346, 158328, 193607, 159200, 377419, 95376, 93906,\
                152146, 360583, 409418, 200161, 131762, 276695, 275144, 55573, 419540, 28365, 306108, 94745, 54410, 279074, \
                386, 66232, 142717, 67953, 280279, 290399, 314668, 213319, 88118, 49277, 135055, 94061, 371359, 47908, 64974, 29402, 299446, \
                160344, 149955, 394989, 195461, 421856, 125270, 220123, 211581, 206307, 217554, 64055, 134332, 310923, 111938, 414335, 689, \
                26579, 96902, 372354, 284544, 187139, 52669, 54037, 401928, 22314, 120554, 158260, 300527, 397637, 105961, 153493, 67881, \
                289531, 297060, 175840, 134621, 245321, 201292, 251323, 96696, 103413, 205748, 10257, 14357, 271090, 330901, 315064, 316551, \
                325705, 401049, 97165, 348159, 56391, 196305, 353599, 155990, 325199, 151208, 371920, 139879, 155948, 191981, 140650, 249067, \
                349029, 98020, 124206, 313345, 1147, 45731, 299926, 122284, 236428, 83394, 47271, 402361, 282338, 156151, 7966, 155776, 293335, \
                85047, 233381, 118662, 85436, 379795, 39983, 366467, 230902, 231011, 226568, 174315, 370161, 168082, 117489, 265184, 207193, \
                264014, 323106, 10762, 55522, 82166, 391623, 192361, 83337, 9357, 348496, 153133, 94213, 50993, 45359, 395434, 62370, 25283, 104371, \
                403180, 291538, 317484, 96083, 372184, 411080, 362944, 66537, 415315, 130033, 72238, 270560, 267251, 355438, 400573, 145847, 208923, \
                403458, 382221, 38848, 244609, 263410, 230677, 264269, 154298, 229297, 88404, 102997, 143260, 189345, 358135, 289845, 133838, 399762, \
                232755, 288849, 51829, 205499, 230553, 296188, 52924, 195553, 410509, 81883, 311368, 302360, 91380, 414844, 295911, 19813, 35160, \
                86950, 58522, 317350, 283516, 97310, 56792, 315758, 290571, 93019, 361446, 359067, 406401, 317379, 235973, 287006, 5051, 206378, \
                185781, 52742, 50331, 335345, 301114, 222768, 166080, 262907, 15756, 361041, 87467, 144179, 372240, 18859, 416591, 373260, 112952,\
                281774, 35376, 74118, 279632, 351142, 232129, 243901, 34061, 207240, 83357, 295912, 380333, 159638, 152914, 307135, 124636, 337864,\
                421943, 143619, 149103, 196359, 151537, 146013, 15819, 78325, 78075, 182333, 306145, 1140, 51402, 65119, 273090, 325797, 124796, \
                116360, 375442, 218221, 394445, 157046, 164146, 247872, 133356, 234817, 220909, 152617, 296270, 276389, 7761, 336921, 342426, 385312, \
                295288, 224133, 256145, 116348, 86816, 36670, 281652, 82868, 325386, 184716, 205217, 124924, 373281, 117940, 22922, 201434, 249420, \
                107518, 96208, 58164, 422147, 60711, 15826, 289370, 247085, 410095, 217170, 107977, 313414, 224735, 133620, 293089, 9617, 36163, \
                276465, 223415, 211923, 1597, 373259, 229326, 317952, 91001, 256683, 3506, 332790, 420254, 306854, 51515, 167081, 278197, 161557, \
                98538, 267096, 417483, 375611, 120311, 286435, 362701, 6495, 230227, 178674, 363449, 317864, 69776, 189588, 95479, 277632, 74578, \
                229340, 272398, 356040, 298282, 397186, 149141, 89577, 44514, 49825, 29619, 380888, 409931, 331867, 201191, 118709, 348862, 353307,\
                192434, 108735, 268731, 250396, 262606, 419377, 303499, 206084, 297807, 58480, 145097, 374184, 94180, 132063, 205675, 177374, 10266,\
                63288, 293649, 216239, 248084, 169570, 44218, 117224, 263262, 57311, 200772, 97879, 334729, 195711, 248215, 317872, 79154, 54866, \
                391857, 76786, 63953]

    archs_to_proc = all_archs[args.arch_list_index:args.arch_list_index+args.num_archs]

    for arch_id in archs_to_proc:
        # assemble command string    
        print(os.getcwd())
        print(os.listdir('.'))
        
        command_list = ['python', 'scripts/main.py', '--full', '--algos', f'{args.algos}',\
                        '--common.seed', '36', '--nas.eval.nasbench101.arch_index', f'{arch_id}',\
                        '--nas.eval.trainer.train_top1_acc_threshold', f'{args.train_top1_acc_threshold}',\
                        '--exp-prefix', f'proxynas_{arch_id}', '--datasets', f'{args.datasets}']
        
        print(command_list)
        ret = subprocess.run(command_list)



if __name__ == '__main__':
    main()