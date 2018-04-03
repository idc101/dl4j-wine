function cols = getColumns()

f = fopen("/Users/iain/code/personal/wine-quality/src/main/resources/winequality-red.csv");
header = fgetl (f);
cols = strsplit (header, ";");
fclose(f);

end