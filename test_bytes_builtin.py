"""
Test bytes builtin operations.

Tests for the bytes() builtin function, including:
- bytes() constructor
- bytes literal syntax
- bytes methods
- bytes operations
"""

import pytest


class TestBytesConstructor:
    """Tests for bytes() constructor."""

    def test_empty_bytes(self):
        """Test creating empty bytes."""
        b = bytes()
        assert b == b''
        assert len(b) == 0

    def test_bytes_from_int(self):
        """Test creating bytes of given size filled with zeros."""
        b = bytes(5)
        assert b == b'\x00\x00\x00\x00\x00'
        assert len(b) == 5

    def test_bytes_from_int_zero(self):
        """Test creating zero-length bytes."""
        b = bytes(0)
        assert b == b''

    def test_bytes_from_int_negative(self):
        """Test that negative size raises ValueError."""
        with pytest.raises(ValueError):
            bytes(-1)

    def test_bytes_from_iterable(self):
        """Test creating bytes from iterable of integers."""
        b = bytes([65, 66, 67])
        assert b == b'ABC'

    def test_bytes_from_list(self):
        """Test creating bytes from list."""
        b = bytes([0, 1, 2, 255])
        assert b == b'\x00\x01\x02\xff'

    def test_bytes_from_tuple(self):
        """Test creating bytes from tuple."""
        b = bytes((72, 101, 108, 108, 111))
        assert b == b'Hello'

    def test_bytes_from_bytes(self):
        """Test creating bytes from another bytes object."""
        # Create bytes dynamically to avoid literal interning
        original = bytes([104, 101, 108, 108, 111])  # 'hello'
        b = bytes(original)
        assert b == b'hello'
        # Note: In CPython, bytes literals may be interned, so we only check equality

    def test_bytes_from_bytearray(self):
        """Test creating bytes from bytearray."""
        ba = bytearray(b'world')
        b = bytes(ba)
        assert b == b'world'

    def test_bytes_from_string_with_encoding(self):
        """Test creating bytes from string with encoding."""
        b = bytes('hello', 'utf-8')
        assert b == b'hello'

    def test_bytes_from_string_ascii(self):
        """Test creating bytes from string with ASCII encoding."""
        b = bytes('test', 'ascii')
        assert b == b'test'

    def test_bytes_from_unicode(self):
        """Test creating bytes from Unicode string."""
        b = bytes('你好', 'utf-8')
        assert len(b) == 6  # 3 bytes per Chinese character in UTF-8

    def test_bytes_invalid_value_in_iterable(self):
        """Test that invalid values in iterable raise ValueError."""
        with pytest.raises(ValueError):
            bytes([65, 256])  # 256 is out of range

    def test_bytes_negative_in_iterable(self):
        """Test that negative values in iterable raise ValueError."""
        with pytest.raises(ValueError):
            bytes([65, -1])


class TestBytesLiteral:
    """Tests for bytes literal syntax."""

    def test_bytes_literal_empty(self):
        """Test empty bytes literal."""
        b = b''
        assert b == bytes()
        assert len(b) == 0

    def test_bytes_literal_ascii(self):
        """Test ASCII bytes literal."""
        b = b'hello'
        assert b[0] == 104  # 'h'
        assert len(b) == 5

    def test_bytes_literal_escape(self):
        """Test bytes literal with escape sequences."""
        b = b'\x00\x01\x02'
        assert b == bytes([0, 1, 2])

    def test_bytes_literal_newline(self):
        """Test bytes literal with newline."""
        b = b'line1\nline2'
        assert b.count(b'\n') == 1

    def test_bytes_literal_mixed(self):
        """Test bytes literal with mixed content."""
        b = b'abc\x00def'
        assert len(b) == 7


class TestBytesOperations:
    """Tests for bytes operations."""

    def test_bytes_concatenation(self):
        """Test bytes concatenation with +."""
        b1 = b'hello'
        b2 = b'world'
        result = b1 + b' ' + b2
        assert result == b'hello world'

    def test_bytes_repetition(self):
        """Test bytes repetition with *."""
        b = b'ab'
        result = b * 3
        assert result == b'ababab'

    def test_bytes_repetition_zero(self):
        """Test bytes repetition with zero."""
        b = b'test'
        result = b * 0
        assert result == b''

    def test_bytes_indexing(self):
        """Test bytes indexing."""
        b = b'hello'
        assert b[0] == 104  # 'h' as integer
        assert b[-1] == 111  # 'o'

    def test_bytes_slicing(self):
        """Test bytes slicing."""
        b = b'hello world'
        assert b[0:5] == b'hello'
        assert b[6:] == b'world'
        assert b[-5:] == b'world'

    def test_bytes_contains(self):
        """Test 'in' operator with bytes."""
        b = b'hello world'
        assert b'world' in b
        assert b'foo' not in b

    def test_bytes_comparison(self):
        """Test bytes comparison."""
        b1 = b'abc'
        b2 = b'abc'
        b3 = b'def'
        assert b1 == b2
        assert b1 != b3
        assert b1 < b3

    def test_bytes_len(self):
        """Test len() on bytes."""
        b = b'hello'
        assert len(b) == 5


class TestBytesMethods:
    """Tests for bytes methods."""

    def test_bytes_upper(self):
        """Test bytes.upper() method."""
        b = b'hello'
        assert b.upper() == b'HELLO'

    def test_bytes_lower(self):
        """Test bytes.lower() method."""
        b = b'HELLO'
        assert b.lower() == b'hello'

    def test_bytes_strip(self):
        """Test bytes.strip() method."""
        b = b'  hello  '
        assert b.strip() == b'hello'

    def test_bytes_lstrip(self):
        """Test bytes.lstrip() method."""
        b = b'  hello  '
        assert b.lstrip() == b'hello  '

    def test_bytes_rstrip(self):
        """Test bytes.rstrip() method."""
        b = b'  hello  '
        assert b.rstrip() == b'  hello'

    def test_bytes_split(self):
        """Test bytes.split() method."""
        b = b'hello world test'
        parts = b.split()
        assert parts == [b'hello', b'world', b'test']

    def test_bytes_split_with_separator(self):
        """Test bytes.split() with separator."""
        b = b'a,b,c'
        parts = b.split(b',')
        assert parts == [b'a', b'b', b'c']

    def test_bytes_join(self):
        """Test bytes.join() method."""
        parts = [b'a', b'b', b'c']
        result = b','.join(parts)
        assert result == b'a,b,c'

    def test_bytes_replace(self):
        """Test bytes.replace() method."""
        b = b'hello world'
        result = b.replace(b'world', b'test')
        assert result == b'hello test'

    def test_bytes_find(self):
        """Test bytes.find() method."""
        b = b'hello world'
        assert b.find(b'world') == 6
        assert b.find(b'foo') == -1

    def test_bytes_count(self):
        """Test bytes.count() method."""
        b = b'hello hello hello'
        assert b.count(b'hello') == 3

    def test_bytes_startswith(self):
        """Test bytes.startswith() method."""
        b = b'hello world'
        assert b.startswith(b'hello') is True
        assert b.startswith(b'world') is False

    def test_bytes_endswith(self):
        """Test bytes.endswith() method."""
        b = b'hello world'
        assert b.endswith(b'world') is True
        assert b.endswith(b'hello') is False

    def test_bytes_hex(self):
        """Test bytes.hex() method."""
        b = b'\x00\x01\x02\xff'
        assert b.hex() == '000102ff'

    def test_bytes_fromhex(self):
        """Test bytes.fromhex() class method."""
        b = bytes.fromhex('48656c6c6f')
        assert b == b'Hello'


class TestBytesImmutable:
    """Tests demonstrating bytes immutability."""

    def test_bytes_immutable(self):
        """Test that bytes objects are immutable."""
        b = b'hello'
        with pytest.raises(TypeError):
            b[0] = 72  # Cannot modify bytes

    def test_bytes_hash(self):
        """Test that bytes can be hashed (required for immutability)."""
        b = b'hello'
        # Should not raise an error
        h = hash(b)
        assert isinstance(h, int)


class TestBytesEdgeCases:
    """Edge case tests for bytes."""

    def test_bytes_all_zeros(self):
        """Test bytes of all zeros."""
        b = bytes(1000)
        assert all(x == 0 for x in b)

    def test_bytes_all_ones(self):
        """Test bytes of all 255s."""
        b = bytes([255] * 100)
        assert all(x == 255 for x in b)

    def test_bytes_large(self):
        """Test creating large bytes object."""
        size = 10 * 1024 * 1024  # 10 MB
        b = bytes(size)
        assert len(b) == size

    def test_bytes_from_generator(self):
        """Test creating bytes from generator expression."""
        b = bytes(x for x in range(256))
        assert len(b) == 256
        assert b[0] == 0
        assert b[255] == 255

    def test_bytes_repr(self):
        """Test bytes representation."""
        b = b'hello'
        r = repr(b)
        assert r == "b'hello'"

    def test_bytes_str(self):
        """Test bytes string representation."""
        b = b'hello'
        s = str(b)  # Returns repr in Python 3
        assert 'hello' in s


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
