# Generate your own bias data file with the the bias data here
data = []
output_file = ''


with open(output_file, 'w') as f:
    for num in data:
        # Convert to 16-bit signed binary representation
        if num < 0:
            num = (1 << 16) + num  # Convert negative value to 16-bit signed
            
        binary = bin(num)[2:].zfill(16)
        
        # Write the binary representation to file
        f.write(binary + '\n')
