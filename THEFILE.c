/******************************************************************************

                            Online C Compiler.
                Code, Compile, Run and Debug C program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <cs50.h>

int company(long number)
{

    long i = number;
    int length = 0;
    while (i > 0) {
        i = i / 10;
        length++;
    }
    long x = number;
    while (x > 99) {
        x = x / 10;
    }
    
    if (length == 15 && (x == 34 || x == 37)) {
        printf("AMERICAN EXPRESS");
    }
    if (length == 16 && (x == 51 || x == 52 || x == 53 || x == 54 || x == 55)) {
        printf("MASTERCARD");
    }    
    if ((length == 13 || length == 16) && (x > 39 && x < 50))
        printf("VISA");
    return 0;
}

int valid(long number)
{
    long i = number;
    int length = 0;
    while (i > 0) {
        i = i / 10;
        length++;
    }
    if (length)
    return 0;
}

int main()
{
    long number = get_long("Enter card number: ");
    valid(number);
    return 0;
}
