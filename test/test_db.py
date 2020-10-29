from unittest import TestCase


from fluxture.db import Database, default, ForeignKey, Model, primary_key


class Person(Model):
    name: primary_key(str)
    age: int


class TestDatabase(TestCase):
    def test_create_table(self):
        db = Database()
        table = db.create_table(Person)
        self.assertEqual(len(table), 0)
        person = Person(name="Foo", age=1337)
        table.append(person)
        self.assertEqual(len(table), 1)
        retrieved_person = next(iter(table))
        self.assertIsInstance(retrieved_person, Person)
        self.assertEqual(retrieved_person, person)
        self.assertEqual(next(iter(table.select(age=1337))), person)
        self.assertCountEqual(table.select(age=0), ())

    def test_define_db(self):
        class TestDB(Database):
            people: Person

        db = TestDB()
        person_table = db[Person]
        self.assertEqual(len(person_table), 0)

    def test_primary_key(self):
        self.assertEqual(Person.primary_key_name, "name")

        def invalid_table():
            class InvalidTable(Model):
                not_primary_key: int

            InvalidTable(1)

        self.assertRaises(TypeError, invalid_table)

    def test_default(self):
        class Number(Model):
            n: default(primary_key(int), 1)

        class TestDB(Database):
            numbers: Number

        db = TestDB()
        numbers_table = db[Number]
        numbers_table.append(Number())
        self.assertEqual(next(iter(numbers_table)), Number(1))

    def test_foreign_key(self):
        class Height(Model):
            person: primary_key(ForeignKey[Person])
            height: int

        class TestDB(Database):
            people: Person
            heights: Height

        db = TestDB()
        person = Person(name="Foo", age=1337)
        db[Person].append(person)
        db[Height].append(Height(person="Foo", height=80))
        h = next(iter(db[Height]))
        self.assertEqual(h.person, person)
