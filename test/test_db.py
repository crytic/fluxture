from unittest import TestCase


from fluxture.db import Database, default, Model, primary_key, unique


class Person(Model):
    name: str
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
