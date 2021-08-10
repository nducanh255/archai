python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,2,2,8,8 --d_model 64 --d_head 32,16,64,16,4 --d_inner 847,951,514,1863,1736 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,2,4,4,2,8,8,4 --d_model 128 --d_head 64,64,64,32,32,32,16,16 --d_inner 1031,1213,1576,1625,666,1684,1277,923 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,8,4,2,2,8 --d_model 128 --d_head 8,32,16,128,32,8 --d_inner 1885,1861,1171,930,886,1604 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,4,4,8,4,2 --d_model 128 --d_head 16,64,16,16,16,64 --d_inner 551,1416,1599,1507,1105,1726 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,4,8,8,8 --d_model 64 --d_head 16,16,8,16,4 --d_inner 1027,1445,1755,1478,645 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,8,8,2,2,2,2 --d_model 64 --d_head 8,4,4,16,32,32,64 --d_inner 1436,1132,513,1764,1675,719,1488 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,4,8,4,2 --d_model 64 --d_head 8,16,16,16,16 --d_inner 658,673,1273,793,1949 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,2,8,8,2 --d_model 512 --d_head 128,512,64,64,128 --d_inner 1454,1687,1943,1768,1305 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,2,4,4,4,8,8 --d_model 64 --d_head 16,64,8,16,16,4,16 --d_inner 858,1517,539,1690,1435,1477,1423 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,8,8,2,2 --d_model 512 --d_head 128,64,64,256,256 --d_inner 1211,1186,1879,1905,1455 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,2,8,8,2,8,8,2 --d_model 512 --d_head 256,256,32,64,128,64,64,512 --d_inner 1755,1939,1467,1318,1526,1527,1721,1131 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,8,4,4,8,4,2 --d_model 64 --d_head 32,16,16,32,4,32,64 --d_inner 1903,571,1229,1618,1901,952,1496 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,4,8,4,2 --d_model 128 --d_head 32,32,32,16,64 --d_inner 987,1152,1119,1991,1163 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,4,2,2,4,8,4 --d_model 512 --d_head 256,64,256,256,256,128,64 --d_inner 1576,1147,1624,1738,2029,1647,1049 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,8,2,8,2 --d_model 64 --d_head 4,8,64,16,32 --d_inner 1332,1889,1375,2042,1368 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,2,8,8,2,4 --d_model 256 --d_head 256,64,256,64,32,256,32 --d_inner 1218,559,666,1325,799,889,1052 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,4,4,2,8,2,8 --d_model 64 --d_head 32,16,8,32,16,16,8 --d_inner 1813,1555,522,1430,629,531,2028 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,8,2,8,4,4,2 --d_model 512 --d_head 128,128,256,128,64,256,128 --d_inner 1923,1334,1596,1351,1339,1100,1176 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,8,4,2,2 --d_model 64 --d_head 4,4,16,16,64 --d_inner 864,1055,1949,1236,1618 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,4,2,4,4,4 --d_model 64 --d_head 8,16,32,8,32,32 --d_inner 1507,733,1609,883,1647,861 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,8,8,8,2,4,4 --d_model 512 --d_head 256,128,128,32,128,256,64 --d_inner 1522,1713,1439,1747,1171,1419,1177 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,2,2,4,4 --d_model 64 --d_head 64,8,16,16,8,16 --d_inner 856,1321,1850,927,1066,1366 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 8,8,4,2,8,2,8,2 --d_model 512 --d_head 128,128,256,128,64,512,32,512 --d_inner 1105,1183,1899,1182,2000,1467,1070,1846 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,4,8,2,4,2,8 --d_model 128 --d_head 64,16,16,128,32,32,8 --d_inner 1017,1927,1892,1055,1577,1074,911 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,2,2,2,4 --d_model 256 --d_head 64,64,128,64,128 --d_inner 773,1356,542,1240,548 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,4,4,4,2,8,8 --d_model 512 --d_head 512,128,128,64,128,64,64 --d_inner 1163,1481,1361,1609,1233,1068,1616 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,4,8,4,8,4 --d_model 64 --d_head 64,16,32,16,16,4,16 --d_inner 1817,1719,1128,1377,1989,783,1087 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,8,8,8,8 --d_model 64 --d_head 64,16,4,16,8 --d_inner 670,2042,1308,773,1724 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,8,4,8,8 --d_model 128 --d_head 128,16,64,16,32 --d_inner 935,1631,1699,536,1802 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,4,4,2,2,8 --d_model 512 --d_head 32,128,64,512,256,32 --d_inner 1746,1026,1717,1857,1938,1834 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,4,2,8,2 --d_model 256 --d_head 64,128,64,64,256 --d_inner 761,932,1622,1488,1671 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,4,8,2,2,2 --d_model 512 --d_head 128,64,64,256,512,512 --d_inner 1807,1276,1881,1798,2000,1432 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,8,2,4,8 --d_model 512 --d_head 256,64,128,512,64,64 --d_inner 1498,1423,1945,1859,1111,1276 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,4,8,8,8,2,2,2 --d_model 512 --d_head 128,256,32,64,32,512,128,128 --d_inner 2043,1584,1716,1608,1248,1464,1955,1451 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,4,4,8,8,2,2 --d_model 512 --d_head 128,64,256,128,128,256,128 --d_inner 1954,1520,1576,1329,1939,1588,1416 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,2,4,2,8 --d_model 512 --d_head 32,512,128,128,32 --d_inner 1500,1040,1233,1522,1144 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,2,8,2,4 --d_model 512 --d_head 512,256,128,512,64 --d_inner 1027,1677,1086,1203,1339 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,2,8,8,4 --d_model 256 --d_head 16,64,64,16,32 --d_inner 910,645,1858,1568,924 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,4,4,4,2,8,8,2 --d_model 128 --d_head 32,32,64,64,32,8,32,32 --d_inner 745,1749,1933,1745,1679,881,666,1867 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,4,2,4,2,8 --d_model 128 --d_head 32,64,32,64,32,32 --d_inner 1568,1321,611,1255,1982,1876 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,4,4,8,8,8 --d_model 512 --d_head 32,256,128,128,32,128 --d_inner 1333,1934,1251,1067,1498,1173 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,8,2,8,4,8 --d_model 64 --d_head 16,4,32,16,16,4 --d_inner 1745,1447,1524,615,1257,1910 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,2,8,2,2,2 --d_model 512 --d_head 32,128,128,128,128,256 --d_inner 1555,1627,1076,1385,1218,1549 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,2,4,8,2,4,2 --d_model 512 --d_head 64,128,64,128,512,128,256 --d_inner 1540,1999,1097,1467,1297,1461,1416 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,2,2,4,4,8,2 --d_model 256 --d_head 32,64,256,32,64,64,128 --d_inner 841,1578,634,1707,870,1604,1709 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,8,4,2,2 --d_model 128 --d_head 8,32,32,64,64 --d_inner 1684,1482,816,1221,723 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,2,2,8,4,2 --d_model 512 --d_head 256,128,128,64,256,512 --d_inner 1305,1829,1791,1978,1979,2018 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,2,2,4,2,2,2,4 --d_model 64 --d_head 8,16,32,16,16,32,64,32 --d_inner 756,1563,1381,1403,1330,753,727,536 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,2,4,2,4,4 --d_model 128 --d_head 16,128,32,32,64,16 --d_inner 666,557,1023,968,1251,1686 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,2,2,4,4,4,2,4 --d_model 64 --d_head 32,32,32,32,8,8,64,32 --d_inner 1788,1856,1065,921,652,846,1761,1551 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 8,2,8,8,4,8,4,4 --d_model 64 --d_head 8,64,4,4,8,16,16,16 --d_inner 1640,1067,965,2035,1698,1152,529,1594 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,8,8,8,8,4 --d_model 64 --d_head 16,8,8,8,4,16 --d_inner 1898,941,1574,1876,1874,1737 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,4,8,8,4,8 --d_model 256 --d_head 128,128,64,64,64,64 --d_inner 1382,796,767,1859,1744,1635 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,2,4,2,4 --d_model 512 --d_head 64,512,256,256,256 --d_inner 1050,1553,1453,1683,1330 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,8,8,2,4 --d_model 128 --d_head 64,8,8,16,32,16 --d_inner 1396,1927,1275,1760,1522,949 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,8,2,2,4,2,4 --d_model 128 --d_head 32,16,64,32,16,128,32 --d_inner 1209,1736,552,1469,2041,1694,539 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,4,4,8,2,2 --d_model 128 --d_head 64,64,32,64,32,128,64 --d_inner 609,1556,1660,1561,1540,1982,808 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,4,4,4,4,4 --d_model 256 --d_head 128,64,128,128,64,64,32 --d_inner 662,1206,1198,992,907,1490,1847 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,4,2,4,8 --d_model 256 --d_head 256,128,256,128,16 --d_inner 531,562,575,1116,1864 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,8,4,8,4 --d_model 256 --d_head 16,32,128,16,64 --d_inner 1782,1239,2015,910,1462 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,4,4,2,4,4,8,8 --d_model 256 --d_head 128,128,64,64,128,32,32,32 --d_inner 1818,602,747,1987,1207,1423,1440,1679 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,8,2,2,4 --d_model 128 --d_head 64,8,32,64,32,64 --d_inner 1448,1902,1647,1665,1928,741 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,4,4,8,4,4,2,2 --d_model 256 --d_head 128,64,32,32,128,64,128,128 --d_inner 1213,1442,1042,1249,822,1114,1263,660 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,2,4,4,2,2 --d_model 128 --d_head 8,128,32,16,64,32 --d_inner 1088,1957,1569,875,1392,602 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,2,2,8,2 --d_model 512 --d_head 256,256,128,128,128 --d_inner 1632,2042,1766,1151,1886 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,2,8,8,2 --d_model 128 --d_head 16,64,16,32,128 --d_inner 1820,1958,592,1144,1605 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,4,2,4,2 --d_model 128 --d_head 128,32,32,16,128 --d_inner 1822,888,1920,1164,676 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,2,4,8,2 --d_model 256 --d_head 128,16,128,64,64,64 --d_inner 610,1766,625,1503,1517,1138 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,2,8,2,8,8,4,4 --d_model 64 --d_head 32,32,16,32,4,16,8,32 --d_inner 563,1149,1160,1422,589,1429,1975,1054 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 8,2,2,4,8,8,4,8 --d_model 256 --d_head 16,128,64,32,32,64,64,16 --d_inner 1302,655,1487,986,1708,1508,829,1501 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,8,8,8,4,2 --d_model 64 --d_head 32,8,8,8,8,16 --d_inner 1295,1876,1199,1933,632,806 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,4,8,4,4,2,4 --d_model 64 --d_head 32,16,8,16,32,32,8 --d_inner 565,1464,1998,1479,1564,533,1880 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,8,4,4,2,4 --d_model 256 --d_head 32,64,32,32,64,32 --d_inner 1021,610,872,1208,1027,1073 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,8,2,8,2,8 --d_model 128 --d_head 64,64,8,32,16,32,32 --d_inner 1981,1481,933,1318,1889,1631,745 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,8,8,4,2,2 --d_model 256 --d_head 32,32,64,128,128,64 --d_inner 558,1162,1011,1590,1328,1260 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,2,8,8,2,4,2,2 --d_model 256 --d_head 128,128,64,16,64,128,256,128 --d_inner 1305,1174,1032,1041,1837,663,1545,960 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,2,2,8,8 --d_model 512 --d_head 512,512,128,128,128 --d_inner 1883,1412,1950,1973,1652 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,4,4,8,2,4,2,8 --d_model 512 --d_head 256,256,64,128,512,64,256,32 --d_inner 2002,1464,1874,1996,1866,1709,1827,1560 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,2,4,4,4,4,4 --d_model 512 --d_head 128,256,256,256,64,64,64 --d_inner 1723,1130,1149,1135,1199,1305,1840 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,8,4,2,2,4,4,8 --d_model 512 --d_head 256,64,64,128,128,256,128,64 --d_inner 1294,1154,1182,1724,1578,1838,1565,1123 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,8,8,2,8,4 --d_model 64 --d_head 8,4,8,64,4,32 --d_inner 1478,1694,1654,1507,598,1356 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,4,2,2,4,4 --d_model 128 --d_head 32,16,32,64,32,64 --d_inner 936,1379,1197,616,1873,1527 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,4,2,4,8,4 --d_model 128 --d_head 16,32,32,64,32,16 --d_inner 1369,1908,1945,1367,1135,869 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,2,8,2,8,2,2 --d_model 128 --d_head 16,64,32,128,32,128,128 --d_inner 1489,780,1195,1913,1137,1025,914 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,2,8,4,8,4,8,8 --d_model 256 --d_head 32,64,32,32,32,64,64,64 --d_inner 1037,752,809,897,1143,818,998,2026 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,8,4,4,8 --d_model 512 --d_head 64,32,256,64,64 --d_inner 1545,1948,1184,2028,1862 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,8,8,2,4,2 --d_model 512 --d_head 512,512,32,128,128,256,256 --d_inner 1983,1208,1272,1649,1417,2037,1165 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,2,2,2,4,8,8 --d_model 128 --d_head 16,128,128,64,32,32,16 --d_inner 1082,761,659,1829,920,1029,550 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 8,8,4,2,2,2,2,4 --d_model 128 --d_head 16,32,64,32,128,64,128,32 --d_inner 1835,1676,1845,1429,1232,1675,718,1127 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,2,8,4,2,2,8,8 --d_model 128 --d_head 128,64,8,64,64,64,16,8 --d_inner 632,1198,1830,1061,1181,1858,1790,1065 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,4,4,4,2 --d_model 512 --d_head 128,128,128,128,128,256 --d_inner 1731,1958,1995,1370,1602,1081 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,8,8,2,4,4,8 --d_model 128 --d_head 32,32,16,64,64,64,32 --d_inner 608,1697,842,1947,1081,730,1189 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,2,2,4,8,8,8,8 --d_model 256 --d_head 64,256,128,64,32,64,64,16 --d_inner 887,1413,1980,1214,1922,1634,2030,776 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,2,4,8,2 --d_model 64 --d_head 64,8,64,8,16,32 --d_inner 1436,1828,952,2028,860,1685 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,4,4,2,8,4,8 --d_model 64 --d_head 32,16,16,32,4,32,8 --d_inner 1033,1288,1814,1036,1365,845,815 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,4,8,4,2 --d_model 256 --d_head 128,16,128,32,64,64 --d_inner 1517,1868,652,576,950,1643 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,2,8,8,2,8,2,4 --d_model 128 --d_head 128,64,32,8,32,32,64,16 --d_inner 649,1122,1748,709,1907,519,571,564 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 8,8,8,2,2,4,8,2 --d_model 256 --d_head 16,16,32,128,64,64,32,256 --d_inner 947,647,882,1907,1097,1264,744,1545 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,2,8,4,2,4 --d_model 128 --d_head 8,64,8,64,32,32 --d_inner 1942,1351,1355,1247,1252,1669 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,2,2,4,8,8 --d_model 256 --d_head 16,64,256,32,64,16 --d_inner 739,547,1482,1434,1311,692 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
