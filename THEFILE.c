/******************************************************************************

                            Online C Compiler.
                Code, Compile, Run and Debug C program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <cs50.h>

int valid(long number)
{
    if (number % 2 == 0) {
        long i = number
        int length = 0
        while (i > 0) {
            i = i / 10
            length++
        }
        
    }
    return 0;
}

int company()
{
    
}

int main()
{
    long number = get_long("Enter card number: ");
    valid(number);
    return 0;
}
