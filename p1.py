people = {

        'Alice': {
            'phone': '2341',
            'addr': 'Foo drive 23'
        },

        'Beth': {
            'phone': '9012',
            'addr': 'Bar street 42'
        },

        'Cecil': {
            'phone': '3158',
            'addr': 'Baz avenue 90'
         }
}

labels = {
        'phone': 'phone number',
        'addr': 'address'
        }
name = input('Name:')

request = input('phone number (p) or address (a)? :')

if request == 'p':
    key = 'phone'
elif request == 'a':
        key = 'addr'
else:
         key = 'none'

if name in people:
    print("%s's %s is %s." %(name, labels[key],people[name][key]))
else:
    print("The name you enter is not exit in the database. you key is %s"%(key))
