#include <stdlib.h>
#include <stdio.h>
#include <time.h>

static int compare(const void* val1, const void* val2){
    const int32_t a = *((int32_t*)val1);
    const int32_t b = *((int32_t*)val2);
    if (a > b) return 1;
    else if (a < b) return -1;
    else return 0;
}

void sortList(int32_t* list, const size_t size){
    qsort((void*)list, size, sizeof(int32_t), compare);
}
