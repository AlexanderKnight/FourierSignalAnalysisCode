import re

newRegex = re.compile('\w+')

newString = newRegex.search('Hello where what dkdkd.dkdk')

print(newString.group())
