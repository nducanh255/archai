python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 8,8,4,8,4,4,8,2 --d_model 256 --d_head 32,32,64,32,64,64,32,128 --d_inner 2042,1012,1999,1922,2036,1440,782,1601 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,8,4,8,2,4,2,4 --d_model 256 --d_head 128,32,64,32,128,64,128,64 --d_inner 1655,1259,1960,867,834,2018,1294,1388 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,8,8,8,2,8 --d_model 128 --d_head 64,64,16,16,16,64,16 --d_inner 1485,1815,891,1260,1205,1946,716 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,4,2,4,4 --d_model 128 --d_head 64,16,32,64,32,32 --d_inner 1807,1302,1707,2029,1254,695 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,4,4,8,8,8,4 --d_model 64 --d_head 16,16,16,8,8,8,16 --d_inner 1493,1045,1196,1542,750,1437,1817 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,4,2,4,8 --d_model 512 --d_head 128,128,256,128,64 --d_inner 1497,1172,1275,2015,1696 --d_embed 512 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,8,2,2,2 --d_model 64 --d_head 8,8,32,32,32 --d_inner 1078,1624,1510,910,1792 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,4,4,8,2,4 --d_model 512 --d_head 128,128,128,64,256,128 --d_inner 1902,1387,1982,1782,1337,1844 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,8,8,2,8,8,2 --d_model 512 --d_head 64,64,64,256,64,64,256 --d_inner 1405,2031,1342,1439,1628,1824,1235 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,4,2,8,4,8 --d_model 256 --d_head 128,64,128,32,64,32 --d_inner 1350,1203,754,948,862,1371 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 4 --n_head 8,8,4,8 --d_model 256 --d_head 32,32,64,32 --d_inner 1087,1629,1474,1429 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,8,8,2,4,4,8 --d_model 512 --d_head 128,64,64,256,128,128,64 --d_inner 1128,1260,1946,1909,1395,1136,1500 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,2,4,2,2,2 --d_model 64 --d_head 32,32,16,32,32,32 --d_inner 1973,1065,1470,562,900,1181 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,4,8,2,4,4 --d_model 256 --d_head 64,64,32,128,64,64 --d_inner 764,1281,1210,1428,1321,1101 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,8,4,2,8,4,4 --d_model 128 --d_head 32,16,32,64,16,32,32 --d_inner 1033,545,1724,1951,1425,1833,2010 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 4 --n_head 2,4,8,8 --d_model 64 --d_head 32,16,8,8 --d_inner 1718,1173,1438,673 --d_embed 64 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,2,8,4,4 --d_model 256 --d_head 64,128,32,64,64 --d_inner 1076,524,666,1522,1809 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,4,8,8,4 --d_model 512 --d_head 64,128,64,64,128 --d_inner 1300,1448,1436,1342,1101 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 4 --n_head 2,4,8,4 --d_model 128 --d_head 64,32,16,32 --d_inner 747,519,1164,564 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,2,4,4,2 --d_model 64 --d_head 8,32,16,16,32 --d_inner 1918,1006,1800,844,1482 --d_embed 64 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,4,2,2,2,2,8 --d_model 256 --d_head 32,64,128,128,128,128,32 --d_inner 1390,1469,2044,844,1907,1398,1251 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,8,2,2,2 --d_model 256 --d_head 32,32,128,128,128 --d_inner 1134,556,1796,2011,1094 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,2,2,2,8,2 --d_model 512 --d_head 128,256,256,256,64,256 --d_inner 1353,2024,1858,1765,1980,1716 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,2,4,4,2,8,2 --d_model 128 --d_head 16,64,32,32,64,16,64 --d_inner 1383,988,1543,1165,1560,1829,1658 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,2,2,8,8 --d_model 256 --d_head 64,128,128,32,32 --d_inner 1849,1403,1744,564,1480 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
