def square_numbers(nums):
    result = [] 
    for i in nums:
        result.append(i*i)
    return result 
def square_numbers_g(nums):
    for i in nums:
        yield (i*i)

my_nums = square_numbers([1,2,3,4])
print(my_nums)
your_nums = square_numbers_g([1,2,3,4,5,6,7,8,9])
print(your_nums)
print(next(your_nums))
print(your_nums.__next__())
for i in your_nums:
    print(i)
    break 
for i, v in enumerate(your_nums):
    print(i,v)
    break
for i, v in enumerate(your_nums):
    print(i,v)

