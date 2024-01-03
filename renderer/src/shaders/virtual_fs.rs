use anyhow::Result;
use shared::FastHashMap;

#[derive(Default)]
pub struct VirtualFs {
    nodes: Nodes,
}

impl VirtualFs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_file(&mut self, path: &VirtualPath, contents: String) -> Result<()> {
        let mut components = path.components();
        let file_name = components.read_file_name()?;

        let mut dirs = Vec::new();
        for component in components {
            match component.try_into()? {
                PathComponent::RootDir | PathComponent::CurDir => dirs.clear(),
                PathComponent::ParentDir => {
                    anyhow::ensure!(dirs.pop().is_some(), "parent dir is not accessible")
                }
                PathComponent::Normal(name) => dirs.push(name),
            }
        }

        let mut children = &mut self.nodes;
        for dir in dirs {
            if let Some(Node::File { .. }) = children.get(dir) {
                anyhow::bail!("path component is not a directory: {dir}");
            }

            let entry = children.entry(dir.to_owned()).or_insert_with(|| Node::Dir {
                childern: Default::default(),
            });
            children = match entry {
                Node::Dir { childern } => childern,
                _ => unreachable!(),
            };
        }

        children.insert(
            file_name.to_owned(),
            Node::File {
                contents: contents.into(),
            },
        );
        Ok(())
    }

    pub fn get_file(&self, base: &VirtualPath, path: &VirtualPath) -> Result<Option<ResolvedFile>> {
        #[derive(Clone, Copy)]
        struct Item<'a, 'b> {
            dir_name: &'a str,
            children: &'b Nodes,
        }

        let mut base_components = base.components();

        // Check if base path is a file
        base_components.read_file_name()?;

        // Resolve base path
        let mut dirs = Vec::<Item>::new();
        for component in base_components {
            match component.try_into()? {
                PathComponent::RootDir | PathComponent::CurDir => dirs.clear(),
                PathComponent::ParentDir => {
                    anyhow::ensure!(
                        dirs.pop().is_some(),
                        "parent dir is not accessible for base path"
                    )
                }
                PathComponent::Normal(name) => {
                    let nodes = dirs.last().map(|item| item.children).unwrap_or(&self.nodes);
                    let children = match nodes.get(name) {
                        Some(Node::Dir { childern }) => childern,
                        Some(Node::File { .. }) => {
                            anyhow::bail!("base path component is not a directory: {name}")
                        }
                        None => {
                            anyhow::bail!("base path not found: {base}");
                        }
                    };

                    dirs.push(Item {
                        dir_name: name,
                        children,
                    });
                }
            }
        }

        // Resolve path
        let mut path_components = path.components();

        // Check if path is a file
        let file_name = path_components.read_file_name()?;

        // Resolve relative path
        for component in path_components {
            match component.try_into()? {
                PathComponent::RootDir => dirs.clear(),
                PathComponent::CurDir => {}
                PathComponent::ParentDir => {
                    anyhow::ensure!(dirs.pop().is_some(), "parent dir is not accessible")
                }
                PathComponent::Normal(name) => {
                    let nodes = dirs.last().map(|item| item.children).unwrap_or(&self.nodes);
                    let children = match nodes.get(name) {
                        Some(Node::Dir { childern }) => childern,
                        Some(Node::File { .. }) => {
                            anyhow::bail!("path component is not a directory: {name}")
                        }
                        None => return Ok(None),
                    };

                    dirs.push(Item {
                        dir_name: name,
                        children,
                    });
                }
            }
        }

        let nodes = dirs.last().map(|item| item.children).unwrap_or(&self.nodes);
        match nodes.get(file_name) {
            Some(Node::File { contents }) => {
                let len = 1
                    + dirs
                        .iter()
                        .map(|item| item.dir_name.len() + 1)
                        .sum::<usize>()
                    + file_name.len();

                let mut absolute_path = String::with_capacity(len);
                for item in &dirs {
                    absolute_path.push_str("/");
                    absolute_path.push_str(item.dir_name);
                }
                absolute_path.push_str("/");
                absolute_path.push_str(file_name);

                Ok(Some(ResolvedFile {
                    absolute_path,
                    contents,
                }))
            }
            Some(Node::Dir { .. }) => anyhow::bail!("not a file"),
            None => Ok(None),
        }
    }
}

pub struct ResolvedFile<'a> {
    pub absolute_path: String,
    pub contents: &'a str,
}

enum Node {
    Dir { childern: Nodes },
    File { contents: String },
}

type Nodes = FastHashMap<String, Node>;

#[repr(transparent)]
pub struct VirtualPath {
    inner: str,
}

impl VirtualPath {
    pub fn new<T: AsRef<str> + ?Sized>(path: &T) -> &Self {
        // SAFETY: `VirtualPath` is a transparent wrapper around `str`
        unsafe { &*(path.as_ref() as *const str as *const Self) }
    }

    pub fn root() -> &'static Self {
        Self::new("/")
    }

    pub fn as_str(&self) -> &str {
        &self.inner
    }

    pub fn components(&self) -> PathComponents<'_> {
        let path = self.inner.as_bytes();
        PathComponents {
            path,
            has_physical_root: !path.is_empty() && path[0] == SEPARATOR,
            front: State::StartDir,
            back: State::Body,
        }
    }
}

impl AsRef<VirtualPath> for str {
    #[inline]
    fn as_ref(&self) -> &VirtualPath {
        VirtualPath::new(self)
    }
}

impl std::fmt::Display for VirtualPath {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PathComponent<'a> {
    Normal(&'a str),
    RootDir,
    CurDir,
    ParentDir,
}

#[derive(Clone)]
pub struct PathComponents<'a> {
    path: &'a [u8],
    has_physical_root: bool,
    front: State,
    back: State,
}

impl<'a> PathComponents<'a> {
    pub fn read_file_name(&mut self) -> Result<&'a str> {
        match self.next_back() {
            Some(PathComponent::Normal(name)) => Ok(name),
            _ => anyhow::bail!("path must point to a file"),
        }
    }

    fn finished(&self) -> bool {
        self.front == State::Done || self.back == State::Done || self.front > self.back
    }

    fn include_cur_dir(&self) -> bool {
        if self.has_physical_root {
            return false;
        }
        let mut iter = self.path.iter();
        match (iter.next(), iter.next()) {
            (Some(b'.'), None) => true,
            (Some(b'.'), Some(b)) => *b == SEPARATOR,
            _ => false,
        }
    }

    fn len_before_body(&self) -> usize {
        let root = (self.front == State::StartDir && self.has_physical_root) as usize;
        let cur = (self.front == State::StartDir && self.include_cur_dir()) as usize;
        root + cur
    }

    fn parse_next_component(&self) -> (usize, Option<PathComponent<'a>>) {
        debug_assert!(self.front == State::Body);
        let (extra, comp) = match self.path.iter().position(|&b| b == SEPARATOR) {
            None => (0, self.path),
            Some(i) => (1, &self.path[..i]),
        };
        // SAFETY: `comp` is a valid UTF-8 string, since it is split on a separator
        (comp.len() + extra, unsafe {
            Self::parse_single_component(comp)
        })
    }

    fn parse_next_component_back(&self) -> (usize, Option<PathComponent<'a>>) {
        debug_assert!(self.back == State::Body);
        let start = self.len_before_body();
        let (extra, comp) = match self.path[start..].iter().rposition(|&b| b == SEPARATOR) {
            None => (0, &self.path[start..]),
            Some(i) => (1, &self.path[start + i + 1..]),
        };
        // SAFETY: `comp` is a valid UTF-8 string, since it is split on a separator
        (comp.len() + extra, unsafe {
            Self::parse_single_component(comp)
        })
    }

    /// # Safety
    /// The following must be true:
    /// - `comp` must be a valid UTF-8 string.
    unsafe fn parse_single_component(comp: &'a [u8]) -> Option<PathComponent<'a>> {
        match comp {
            b"" | b"." => None,
            b".." => Some(PathComponent::ParentDir),
            _ => Some(PathComponent::Normal(std::str::from_utf8_unchecked(comp))),
        }
    }
}

impl<'a> Iterator for PathComponents<'a> {
    type Item = PathComponent<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.finished() {
            match self.front {
                State::StartDir => {
                    self.front = State::Body;
                    if self.has_physical_root {
                        debug_assert!(!self.path.is_empty());
                        self.path = &self.path[1..];
                        return Some(PathComponent::RootDir);
                    } else if self.include_cur_dir() {
                        debug_assert!(!self.path.is_empty());
                        self.path = &self.path[1..];
                        return Some(PathComponent::CurDir);
                    }
                }
                State::Body if !self.path.is_empty() => {
                    let (size, comp) = self.parse_next_component();
                    self.path = &self.path[size..];
                    if comp.is_some() {
                        return comp;
                    }
                }
                State::Body => self.front = State::Done,
                State::Done => unreachable!(),
            }
        }
        None
    }
}

impl<'a> DoubleEndedIterator for PathComponents<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while !self.finished() {
            match self.back {
                State::Body if self.path.len() > self.len_before_body() => {
                    let (size, comp) = self.parse_next_component_back();
                    self.path = &self.path[..self.path.len() - size];
                    if comp.is_some() {
                        return comp;
                    }
                }
                State::Body => self.back = State::StartDir,
                State::StartDir => {
                    self.back = State::Done;
                    if self.has_physical_root {
                        self.path = &self.path[..self.path.len() - 1];
                        return Some(PathComponent::RootDir);
                    } else if self.include_cur_dir() {
                        self.path = &self.path[..self.path.len() - 1];
                        return Some(PathComponent::CurDir);
                    }
                }
                State::Done => unreachable!(),
            }
        }
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd)]
enum State {
    StartDir,
    Body,
    Done,
}

const SEPARATOR: u8 = b'/';
