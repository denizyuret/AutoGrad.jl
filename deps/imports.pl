#!/usr/bin/perl -w
use strict;

my %symbols;
while(<>) {
    if (/^@\S+\s*(\S+?)\(/) {
	$symbols{$1}++;
    }
}

print("import Base: ");
print(join(", ", sort keys %symbols));
print("\n");
