# Generate your own scale data file with the the scale data here
data = []

output_file = ''

with open(output_file, 'w') as f:
    for num in data:
        # Convert to 16-bit binary representation
        binary = bin(num & 0xFFFF)[2:].zfill(16)
        
        # Write the binary representation to file
        f.write(binary + '\n')
