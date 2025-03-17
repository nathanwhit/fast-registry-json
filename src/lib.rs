use std::marker::PhantomData;

use simd::{Simd8, Simd8x64, make_uint8x16_t};

pub mod simd;

pub struct BufBlockReader<'a, const STEP_SIZE: usize> {
    buf: &'a [u8],
    len_minus_step: usize,
    idx: usize,
}

impl<'a, const STEP_SIZE: usize> BufBlockReader<'a, STEP_SIZE> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            len_minus_step: buf.len().saturating_sub(STEP_SIZE),
            idx: 0,
            buf,
        }
    }

    pub fn block_index(&self) -> usize {
        self.idx
    }

    pub fn has_full_block(&self) -> bool {
        self.idx < self.len_minus_step
    }

    pub fn full_block(&self) -> &'a [u8] {
        &self.buf[self.idx..]
    }

    pub fn get_remainder(&self, dest: &mut [u8]) -> usize {
        if self.buf.len() == self.idx {
            return 0;
        }

        dest[..STEP_SIZE].fill(0x20);
        let remainder = &self.buf[self.idx..];
        dest[..remainder.len()].copy_from_slice(remainder);
        self.buf.len() - self.idx
    }

    pub fn advance(&mut self) {
        self.idx += STEP_SIZE;
    }
}

#[derive(Default)]
struct JsonEscapeScanner {
    next_is_escaped: u64,
}

struct EscapedAndEscape {
    /**
     * Mask of escaped characters.
     *
     * ```ignore
     * \n \\n \\\n \\\\n \
     * 0100100010100101000
     *  n  \   \ n  \ \
     * ```
     */
    escaped: u64,
    /**
     * Mask of escape characters.
     *
     * ```ignore
     * \n \\n \\\n \\\\n \
     * 1001000101001010001
     * \  \   \ \  \ \   \
     * ```
     */
    escape: u64,
}

impl JsonEscapeScanner {
    fn new() -> Self {
        Self::default()
    }

    /**
     * Get a mask of both escape and escaped characters (the characters following a backslash).
     *
     * @param potential_escape A mask of the character that can escape others (but could be
     *        escaped itself). e.g. block.eq('\\')
     */
    fn next(&mut self, backslash: u64) -> EscapedAndEscape {
        // |                                | Mask (shows characters instead of 1's) | Depth | Instructions        |
        // |--------------------------------|----------------------------------------|-------|---------------------|
        // | string                         | `\\n_\\\n___\\\n___\\\\___\\\\__\\\`   |       |                     |
        // |                                | `    even   odd    even   odd   odd`   |       |                     |
        // | potential_escape               | ` \  \\\    \\\    \\\\   \\\\  \\\`   | 1     | 1 (backslash & ~first_is_escaped)
        // | escape_and_terminal_code       | ` \n \ \n   \ \n   \ \    \ \   \ \`   | 5     | 5 (next_escape_and_terminal_code())
        // | escaped                        | `\    \ n    \ n    \ \    \ \   \ ` X | 6     | 7 (escape_and_terminal_code ^ (potential_escape | first_is_escaped))
        // | escape                         | `    \ \    \ \    \ \    \ \   \ \`   | 6     | 8 (escape_and_terminal_code & backslash)
        // | first_is_escaped               | `\                                 `   | 7 (*) | 9 (escape >> 63) ()
        //                                                                               (*) this is not needed until the next iteration
        let escape_and_terminal_code =
            Self::next_escape_and_terminal_code(backslash & !self.next_is_escaped);
        let escaped = escape_and_terminal_code ^ (backslash | self.next_is_escaped);
        let escape = escape_and_terminal_code & backslash;
        self.next_is_escaped = escape >> 63;
        EscapedAndEscape { escaped, escape }
    }

    /**
     * Returns a mask of the next escape characters (masking out escaped backslashes), along with
     * any non-backslash escape codes.
     *
     * \n \\n \\\n \\\\n returns:
     * \n \   \ \n \ \
     * 11 100 1011 10100
     *
     * You are expected to mask out the first bit yourself if the previous block had a trailing
     * escape.
     *
     * & the result with potential_escape to get just the escape characters.
     * ^ the result with (potential_escape | first_is_escaped) to get escaped characters.
     */
    fn next_escape_and_terminal_code(potential_escape: u64) -> u64 {
        // If we were to just shift and mask out any odd bits, we'd actually get a *half* right answer:
        // any even-aligned backslash runs would be correct! Odd-aligned backslash runs would be
        // inverted (\\\ would be 010 instead of 101).
        //
        // ```
        // string:              | ____\\\\_\\\\_____ |
        // maybe_escaped | ODD  |     \ \   \ \      |
        //               even-aligned ^^^  ^^^^ odd-aligned
        // ```
        //
        // Taking that into account, our basic strategy is:
        //
        // 1. Use subtraction to produce a mask with 1's for even-aligned runs and 0's for
        //    odd-aligned runs.
        // 2. XOR all odd bits, which masks out the odd bits in even-aligned runs, and brings IN the
        //    odd bits in odd-aligned runs.
        // 3. & with backslash to clean up any stray bits.
        // runs are set to 0, and then XORing with "odd":
        //
        // |                                | Mask (shows characters instead of 1's) | Instructions        |
        // |--------------------------------|----------------------------------------|---------------------|
        // | string                         | `\\n_\\\n___\\\n___\\\\___\\\\__\\\`   |
        // |                                | `    even   odd    even   odd   odd`   |
        // | maybe_escaped                  | `  n  \\n    \\n    \\\_   \\\_  \\` X | 1 (potential_escape << 1)
        // | maybe_escaped_and_odd          | ` \n_ \\n _ \\\n_ _ \\\__ _\\\_ \\\`   | 1 (maybe_escaped | odd)
        // | even_series_codes_and_odd      | `  n_\\\  _    n_ _\\\\ _     _    `   | 1 (maybe_escaped_and_odd - potential_escape)
        // | escape_and_terminal_code       | ` \n \ \n   \ \n   \ \    \ \   \ \`   | 1 (^ odd)
        //

        // Escaped characters are characters following an escape.
        let maybe_escaped = potential_escape << 1;

        const ODD_BITS: u64 = 0xAAAAAAAAAAAAAAAA;

        // To distinguish odd from even escape sequences, therefore, we turn on any *starting*
        // escapes that are on an odd byte. (We actually bring in all odd bits, for speed.)
        // - Odd runs of backslashes are 0000, and the code at the end ("n" in \n or \\n) is 1.
        // - Odd runs of backslashes are 1111, and the code at the end ("n" in \n or \\n) is 0.
        // - All other odd bytes are 1, and even bytes are 0.
        let maybe_escaped_and_odd_bits = maybe_escaped | ODD_BITS;
        let even_series_codes_and_odd_bits = maybe_escaped_and_odd_bits - potential_escape;

        // Now we flip all odd bytes back with xor. This:
        // - Makes odd runs of backslashes go from 0000 to 1010
        // - Makes even runs of backslashes go from 1111 to 1010
        // - Sets actually-escaped codes to 1 (the n in \n and \\n: \n = 11, \\n = 100)
        // - Resets all other bytes to 0
        return even_series_codes_and_odd_bits ^ ODD_BITS;
    }
}

/// A block of JSON string processing results
#[derive(Debug)]
pub struct JsonStringBlock {
    // Escaped characters (characters following an escape character)
    escaped: u64,
    // Real (non-backslashed) quotes
    quote: u64,
    // String characters (includes start quote but not end quote)
    in_string: u64,
}

impl JsonStringBlock {
    pub fn new(escaped: u64, quote: u64, in_string: u64) -> Self {
        Self {
            escaped,
            quote,
            in_string,
        }
    }

    // Escaped characters (characters following an escape character)
    pub fn escaped(&self) -> u64 {
        self.escaped
    }

    // Real (non-backslashed) quotes
    pub fn quote(&self) -> u64 {
        self.quote
    }

    // Only characters inside the string (not including the quotes)
    pub fn string_content(&self) -> u64 {
        self.in_string & !self.quote
    }

    // Return a mask of whether the given characters are inside a string (only works on non-quotes)
    pub fn non_quote_inside_string(&self, mask: u64) -> u64 {
        mask & self.in_string
    }

    // Return a mask of whether the given characters are outside a string (only works on non-quotes)
    pub fn non_quote_outside_string(&self, mask: u64) -> u64 {
        mask & !self.in_string
    }

    // Tail of string (everything except the start quote)
    pub fn string_tail(&self) -> u64 {
        self.in_string ^ self.quote
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    UnclosedString,
}

pub struct JsonStringScanner {
    // Scans for escape characters
    escape_scanner: JsonEscapeScanner,
    // Whether the last iteration was still inside a string (all 1's = true, all 0's = false).
    prev_in_string: u64,
}

impl JsonStringScanner {
    pub fn new() -> Self {
        Self {
            escape_scanner: JsonEscapeScanner::new(),
            prev_in_string: 0,
        }
    }

    /// Return a mask of all string characters plus end quotes.
    ///
    /// prev_escaped is overflow saying whether the next character is escaped.
    /// prev_in_string is overflow saying whether we're still in a string.
    ///
    /// Backslash sequences outside of quotes will be detected in stage 2.
    pub fn next(&mut self, input: &simd::Simd8x64<u8>) -> JsonStringBlock {
        let backslash = input.eq(b'\\');
        let escaped = self.escape_scanner.next(backslash).escaped;
        let quote = input.eq(b'"') & !escaped;

        // prefix_xor flips on bits inside the string (and flips off the end quote).
        // Then we xor with prev_in_string: if we were in a string already, its effect is flipped
        // (characters inside strings are outside, and characters outside strings are inside).
        let in_string = prefix_xor(quote) ^ self.prev_in_string;

        // Check if we're still in a string at the end of the box so the next block will know
        self.prev_in_string = (in_string as i64).wrapping_shr(63) as u64;

        JsonStringBlock::new(escaped, quote, in_string)
    }

    /// Returns either UnclosedString or Success
    pub fn finish(&self) -> Result<(), Error> {
        if self.prev_in_string != 0 {
            Err(Error::UnclosedString)
        } else {
            Ok(())
        }
    }
}

/// Performs a prefix XOR operation on a 64-bit value
///
/// This is equivalent to the prefix_xor function in the C++ code
fn prefix_xor(mask: u64) -> u64 {
    let mut result = mask;
    // Prefix XOR each bit with the previous bits
    result ^= result << 1;
    result ^= result << 2;
    result ^= result << 4;
    result ^= result << 8;
    result ^= result << 16;
    result ^= result << 32;
    result
}

/// A block of JSON character classification results
#[derive(Debug)]
pub struct JsonCharacterBlock {
    whitespace: u64,
    op: u64,
}

mod classify {
    use super::*;

    #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
    pub fn classify(input: &simd::Simd8x64<u8>) -> JsonCharacterBlock {
        let table1 = make_uint8x16_t(16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0).into();
        let table2 = make_uint8x16_t(8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0).into();

        let v = simd::Simd8x64::from_chunks([
            (input.chunks[0] & Simd8::<u8>::splat(0xf)).lookup_16_table(table1)
                & (input.chunks[0].shr::<4>()).lookup_16_table(table2),
            (input.chunks[1] & Simd8::<u8>::splat(0xf)).lookup_16_table(table1)
                & (input.chunks[1].shr::<4>()).lookup_16_table(table2),
            (input.chunks[2] & Simd8::<u8>::splat(0xf)).lookup_16_table(table1)
                & (input.chunks[2].shr::<4>()).lookup_16_table(table2),
            (input.chunks[3] & Simd8::<u8>::splat(0xf)).lookup_16_table(table1)
                & (input.chunks[3].shr::<4>()).lookup_16_table(table2),
        ]);

        let op = Simd8x64::from_chunks([
            v.chunks[0].any_bits_set(0x7.into()),
            v.chunks[1].any_bits_set(0x7.into()),
            v.chunks[2].any_bits_set(0x7.into()),
            v.chunks[3].any_bits_set(0x7.into()),
        ])
        .to_bitmask();

        let whitespace = Simd8x64::from_chunks([
            v.chunks[0].any_bits_set(0x18.into()),
            v.chunks[1].any_bits_set(0x18.into()),
            v.chunks[2].any_bits_set(0x18.into()),
            v.chunks[3].any_bits_set(0x18.into()),
        ])
        .to_bitmask();

        JsonCharacterBlock { whitespace, op }
    }
}

impl JsonCharacterBlock {
    /// Classify a block of JSON text
    pub fn classify(input: &simd::Simd8x64<u8>) -> Self {
        #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
        return classify::classify(input);
        #[cfg(not(any(target_arch = "aarch64", target_arch = "arm64ec")))]
        unimplemented!()
    }

    /// Returns a mask of whitespace characters
    pub fn whitespace(&self) -> u64 {
        self.whitespace
    }

    /// Returns a mask of operator characters
    pub fn op(&self) -> u64 {
        self.op
    }

    /// Returns a mask of scalar characters (not whitespace or operators)
    pub fn scalar(&self) -> u64 {
        !(self.op() | self.whitespace())
    }
}

#[derive(Debug)]
/// A block of JSON parsing results combining string and character info
pub struct JsonBlock {
    /// String and escape characters
    pub string: JsonStringBlock,
    /// Whitespace, structural characters ('operators'), scalars
    pub characters: JsonCharacterBlock,
    /// Whether the previous character was a scalar
    follows_potential_nonquote_scalar: u64,
}

impl JsonBlock {
    /// Create a new JsonBlock
    pub fn new(
        string: JsonStringBlock,
        characters: JsonCharacterBlock,
        follows_potential_nonquote_scalar: u64,
    ) -> Self {
        Self {
            string,
            characters,
            follows_potential_nonquote_scalar,
        }
    }

    /// The start of structurals.
    /// In simdjson prior to v0.3, these were called the pseudo-structural characters.
    pub fn structural_start(&self) -> u64 {
        self.potential_structural_start() & !self.string.string_tail()
    }

    /// All JSON whitespace (i.e. not in a string)
    pub fn whitespace(&self) -> u64 {
        self.non_quote_outside_string(self.characters.whitespace())
    }

    /// Whether the given characters are inside a string (only works on non-quotes)
    pub fn non_quote_inside_string(&self, mask: u64) -> u64 {
        self.string.non_quote_inside_string(mask)
    }

    /// Whether the given characters are outside a string (only works on non-quotes)
    pub fn non_quote_outside_string(&self, mask: u64) -> u64 {
        self.string.non_quote_outside_string(mask)
    }

    // ------------ Private methods from C++ implementation ------------

    /// Structural elements ([,],{,},:, comma) plus scalar starts like 123, true and "abc".
    /// They may reside inside a string.
    fn potential_structural_start(&self) -> u64 {
        self.characters.op() | self.potential_scalar_start()
    }

    /// The start of non-operator runs, like 123, true and "abc".
    /// It may reside inside a string.
    fn potential_scalar_start(&self) -> u64 {
        // The term "scalar" refers to anything except structural characters and white space
        // (so letters, numbers, quotes).
        // Whenever it is preceded by something that is not a structural element ({,},[,],:, ") nor a white-space
        // then we know that it is irrelevant structurally.
        self.characters.scalar() & !self.follows_potential_scalar()
    }

    /// Whether the given character is immediately after a non-operator like 123, true.
    /// The characters following a quote are not included.
    fn follows_potential_scalar(&self) -> u64 {
        // follows_potential_nonquote_scalar: is defined as marking any character that follows a character
        // that is not a structural element ({,},[,],:, comma) nor a quote (") and that is not a
        // white space.
        // It is understood that within quoted region, anything at all could be marked (irrelevant).
        self.follows_potential_nonquote_scalar
    }
}

/// Scans JSON for important bits: structural characters or 'operators', strings, and scalars.
///
/// The scanner starts by calculating two distinct things:
/// - string characters (taking \" into account)
/// - structural characters or 'operators' ([]{},:, comma)
///   and scalars (runs of non-operators like 123, true and "abc")
///
/// To minimize data dependency (a key component of the scanner's speed), it finds these in parallel:
/// in particular, the operator/scalar bit will find plenty of things that are actually part of
/// strings. When we're done, JsonBlock will fuse the two together by masking out tokens that are
/// part of a string.
pub struct JsonScanner {
    /// Whether the last character of the previous iteration is part of a scalar token
    /// (anything except whitespace or a structural character/'operator').
    prev_scalar: u64,
    string_scanner: JsonStringScanner,
}

impl JsonScanner {
    pub fn new() -> Self {
        Self {
            prev_scalar: 0,
            string_scanner: JsonStringScanner::new(),
        }
    }

    pub fn next(&mut self, input: &simd::Simd8x64<u8>) -> JsonBlock {
        let strings = self.string_scanner.next(input);
        // Identifies the white-space and the structural characters
        let characters = JsonCharacterBlock::classify(input);

        // The term "scalar" refers to anything except structural characters and white space
        // (so letters, numbers, quotes).
        // We want follows_scalar to mark anything that follows a non-quote scalar (so letters and numbers).
        //
        // A terminal quote should either be followed by a structural character (comma, brace, bracket, colon)
        // or nothing. However, we still want ' "a string"true ' to mark the 't' of 'true' as a potential
        // pseudo-structural character just like we would if we had  ' "a string" true '; otherwise we
        // may need to add an extra check when parsing strings.
        let nonquote_scalar = characters.scalar() & !strings.quote();
        let follows_nonquote_scalar = follows(nonquote_scalar, &mut self.prev_scalar);

        JsonBlock::new(strings, characters, follows_nonquote_scalar)
    }

    pub fn finish(&self) -> Result<(), Error> {
        self.string_scanner.finish()
    }
}

/// Check if the current character immediately follows a matching character.
///
/// For example, this checks for quotes with backslashes in front of them:
///
/// ```ignore
/// let backslashed_quote = in.eq('"') & follows(in.eq('\\'), &mut prev_backslash);
/// ```
fn follows(match_mask: u64, overflow: &mut u64) -> u64 {
    let result = (match_mask << 1) | *overflow;
    *overflow = match_mask >> 63;
    result
}

pub struct BitIndexer<'a> {
    tail: &'a mut [u32],
    idx: usize,
}

fn zero_leading_bit(rev_bits: u64, leading_zeroes: u32) -> u64 {
    rev_bits ^ (0x8000000000000000u64.wrapping_shr(leading_zeroes))
}

impl<'a> BitIndexer<'a> {
    pub fn new(tail: &'a mut [u32]) -> Self {
        Self { tail, idx: 0 }
    }

    pub fn write_index(&mut self, index: u32, rev_bits: &mut u64, i: usize) {
        if *rev_bits == 0 {
            return;
        }
        let lz = rev_bits.leading_zeros();
        unsafe {
            *self.tail.get_unchecked_mut(self.idx + i) = index + lz;
        }
        *rev_bits = zero_leading_bit(*rev_bits, lz);
    }

    pub fn write_indexes(&mut self, index: u32, rev_bits: &mut u64, start: usize) {
        self.write_index(index, rev_bits, start);
        self.write_index(index, rev_bits, start + 1);
        self.write_index(index, rev_bits, start + 2);
        self.write_index(index, rev_bits, start + 3);
    }

    pub fn write_indexes_stepped(
        &mut self,
        index: u32,
        rev_bits: &mut u64,
        cnt: usize,
        start: usize,
        end: usize,
    ) {
        self.write_indexes(index, rev_bits, start);
        if start + 4 < end {
            if start + 4 < cnt {
                self.write_indexes_stepped(index, rev_bits, cnt, start + 4, end);
            }
        }
    }

    pub fn write(&mut self, index: u32, bits: u64) -> usize {
        if bits == 0 {
            return 0;
        }

        let cnt = bits.count_ones();
        let mut rev_bits = bits.reverse_bits();

        self.write_indexes_stepped(index, &mut rev_bits, cnt as usize, 0, 24);

        if 24 < cnt {
            for i in 24..cnt {
                self.write_index(index, &mut rev_bits, i as usize);
            }
        }

        self.idx += cnt as usize;
        cnt as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub start: u32,
    pub end: u32,
    pub kind: TokenKind,
}

impl Token {
    pub fn value<'a>(&self, input: &'a str) -> &'a str {
        &input[self.start as usize..self.end as usize]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Operator,
    String,
    Scalar,
}
pub struct Tokenizer<'a> {
    scanner: JsonScanner,
    tokens: Vec<Token>,
    buf: Vec<u32>,

    num_tokens: usize,

    block_reader: BufBlockReader<'a, 64>,
    idx: u32,

    incomplete_string: bool,

    input: &'a [u8],
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            scanner: JsonScanner::new(),
            tokens: Vec::with_capacity(input.len() / 3),
            buf: vec![0; 64 * 3],
            block_reader: BufBlockReader::new(input),
            idx: 0,
            incomplete_string: false,
            input,
            num_tokens: 0,
        }
    }

    fn process_json_block(&mut self, json_block: JsonBlock, block: &simd::Simd8x64<u8>) {
        let mut bit_indexer = BitIndexer::new(&mut self.buf);

        let ops = json_block.characters.op() & !json_block.string.in_string;
        let mut strings = json_block.string.in_string & !json_block.string.quote;

        let wrote_ops = bit_indexer.write(self.idx, ops);
        let mut bit_indexer = BitIndexer::new(&mut self.buf[wrote_ops..]);

        if self.incomplete_string {
            let pos = 64 - json_block.string.quote.trailing_zeros();
            if pos == 0 {
                return;
            }
            // let pos = pos - 1;
            let last = self.tokens.len() - 1;
            self.tokens[last].end = self.idx + (64 - pos);
            // eprintln!(
            //     "{}",
            //     self.tokens[last].value(unsafe { std::str::from_utf8_unchecked(self.input) })
            // );
            // eprintln!("pos: {}", pos);
            let mask = if pos <= 1 {
                0
            } else {
                (!0u64) << ((64 - pos) + 1)
            };
            self.incomplete_string = false;
            strings = strings & mask;
        }
        let wrote_strings = bit_indexer.write(self.idx, strings);
        // let mut bit_indexer = BitIndexer::new(&mut self.buf[wrote_ops + wrote_strings..]);
        // let wrote_scalars = bit_indexer.write(
        //     self.idx,
        //     json_block.potential_scalar_start() & !strings & !json_block.string.quote,
        // );

        let num_tokens = self.tokens.len();
        for i in 0..wrote_ops {
            let start = self.buf[i as usize];
            let end = start + 1;
            self.tokens.push(Token {
                start,
                end,
                kind: TokenKind::Operator,
            });
        }

        let mut i = wrote_ops;
        let mut last_end = 0;
        while i < wrote_ops + wrote_strings {
            let start = self.buf[i as usize];
            // find the end of the string
            let mut end = start;
            i += 1;
            while i < wrote_ops + wrote_strings && self.buf[i as usize] == end + 1 {
                end = self.buf[i as usize];
                i += 1;
            }
            last_end = end;
            self.tokens.push(Token {
                start,
                end: end + 1,
                kind: TokenKind::String,
            });
        }

        self.tokens[num_tokens..].sort_unstable_by_key(|t| t.start);

        if wrote_strings > 0 {
            let last_mask = 0x8000000000000000;

            if last_end - self.idx == 63
                && (last_mask & strings | json_block.string.quote) != 0
                && (last_mask & json_block.string.quote) == 0
            {
                self.incomplete_string = true;
            }
        }

        // for i in wrote_ops + wrote_strings..wrote_ops + wrote_strings + wrote_scalars {
        //     let start = self.buf[i as usize];
        //     let end = start + 1;
        //     self.tokens.push(Token {
        //         start,
        //         end,
        //         kind: TokenKind::Scalar,
        //     });
        // }
    }

    pub fn tokenize(mut self) -> Result<Vec<Token>, Error> {
        while self.block_reader.has_full_block() {
            let block = self.block_reader.full_block();
            let block = simd::Simd8x64::<u8>::load(arrayref::array_ref![block, 0, 64]);
            let json_block = self.scanner.next(&block);
            self.block_reader.advance();
            self.process_json_block(json_block, &block);
            self.idx += 64;
        }

        let mut remainder_buf = [0; 64];
        let _pad = self.block_reader.get_remainder(&mut remainder_buf);
        let block = simd::Simd8x64::<u8>::load(arrayref::array_ref![&remainder_buf, 0, 64]);
        let json_block = self.scanner.next(&block);
        self.block_reader.advance();
        self.process_json_block(json_block, &block);
        self.idx += 64;
        self.scanner.finish()?;
        Ok(self.tokens)
    }
}

pub fn pluck_versions_from_tokens<'i>(input: &'i str, tokens: &[Token]) -> Vec<&'i str> {
    enum State {
        Start,
        InVersions,
    }
    let mut state = State::Start;
    let mut versions = Vec::new();
    let mut object_depth = 0;
    let input_bytes = input.as_bytes();
    for token in tokens {
        match token.kind {
            TokenKind::String => {
                if object_depth == 1 && token.value(input) == "versions" {
                    state = State::InVersions;
                } else if object_depth == 2 && matches!(state, State::InVersions) {
                    versions.push(token.value(input));
                }
            }
            TokenKind::Operator => {
                let v = input_bytes[token.start as usize];
                if v == b'{' {
                    object_depth += 1;
                } else if v == b'}' {
                    if object_depth == 2 && matches!(state, State::InVersions) {
                        state = State::Start;
                    }
                    object_depth -= 1;
                }
            }
            _ => {}
        }
    }
    versions
}

pub fn pluck_versions(input: &str) -> Vec<&str> {
    let tokenizer = Tokenizer::new(input.as_bytes());
    let tokens = tokenizer.tokenize().unwrap();
    pluck_versions_from_tokens(input, &tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    use pretty_assertions::assert_eq;
    #[test]
    fn zero_leading_bit_works() {
        let cases: &[(u64, u64)] = &[(0b0100, 0), (0b0101, 0b0001)];
        for (rev_bits, expected) in cases {
            let (rev_bits, expected) = (*rev_bits, *expected);
            let leading_zeroes = rev_bits.leading_zeros();
            let result = zero_leading_bit(rev_bits, leading_zeroes);
            assert_eq!(result, expected);
        }
    }

    fn string(start: u32, s: &str) -> Token {
        Token {
            start,
            end: start + s.len() as u32,
            kind: TokenKind::String,
        }
    }

    fn op(start: u32) -> Token {
        Token {
            start,
            end: start + 1,
            kind: TokenKind::Operator,
        }
    }

    struct TokensBuilder {
        tokens: Vec<Token>,
    }

    impl TokensBuilder {
        fn new() -> Self {
            Self { tokens: Vec::new() }
        }

        fn then(self, offset: u32, f: impl FnOnce(Self, u32) -> Self) -> Self {
            let last_end = self.tokens.last().map_or(0, |t| t.end);
            let s = f(self, last_end + offset);
            s
        }

        fn string(mut self, start: u32, s: &str) -> Self {
            self.tokens.push(string(start, s));
            self
        }

        fn op(mut self, start: u32) -> Self {
            self.tokens.push(op(start));
            self
        }

        fn then_string(self, offset: u32, s: &str) -> Self {
            self.then(offset, |b, i| b.string(i, s))
        }

        fn then_op(self, offset: u32) -> Self {
            self.then(offset, |b, i| b.op(i))
        }

        fn build(self) -> Vec<Token> {
            self.tokens
        }
    }

    fn annotated(input: &str, toks: &[Token]) -> String {
        let mut annotated = String::new();
        for tok in toks {
            annotated.push_str(&format!("{} {} {:?}\n", tok.start, tok.end, tok.kind));
            annotated.push('"');
            if tok.start <= tok.end {
                annotated.push_str(&input[tok.start as usize..tok.end as usize]);
            } else {
                annotated.push_str("badbadbad");
            }
            annotated.push('"');
            annotated.push('\n');
        }
        annotated
    }

    fn assert_tokens_eq(input: &str, expected: Vec<Token>) {
        let tokens = Tokenizer::new(input.as_bytes()).tokenize().unwrap();
        if tokens != expected {
            eprintln!("got\n-----\n{}", annotated(input, &tokens));
            eprintln!("expected\n-----\n{}", annotated(input, &expected));
        }
        assert_eq!(tokens, expected);
    }

    #[test]
    fn incomplete_string_works() {
        let input = r#"{"versions":{"aaaaaaaaaaaaaaaaaaaaaaaaaaaaa":{},"bcdefghijkabcd":"asdf"}}"#;

        let expected = TokensBuilder::new()
            .then_op(0)
            .then_string(1, "versions")
            .then_op(1)
            .then_op(0)
            .then_string(1, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            .then_op(1) // :
            .then_op(0) // {
            .then_op(0) // }
            .then_op(0) // ,
            .then_string(1, "bcdefghijkabcd")
            .then_op(1) // :
            .then_string(1, "asdf")
            .then_op(1) // }
            .then_op(0) // }
            .build();
        assert_tokens_eq(input, expected);

        eprintln!("one ok");
        let input = r#"{"versions":{"aaaaaaaaaaaaaaaaaaaaaaaaaaaaa":{},"bcdefghijkab":"asdf"}}"#;
        let expected = TokensBuilder::new()
            .then_op(0)
            .then_string(1, "versions")
            .then_op(1) // :
            .then_op(0) // {
            .then_string(1, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            .then_op(1) // :
            .then_op(0) // {
            .then_op(0) // }
            .then_op(0) // ,
            .then_string(1, "bcdefghijkab")
            .then_op(1) // :
            .then_string(1, "asdf")
            .then_op(1) // }
            .then_op(0) // }
            .build();
        assert_tokens_eq(input, expected);
    }
}
